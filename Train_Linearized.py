import os
from torch._C import device
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, wandb
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from librosa import griffinlim
from scipy.io import wavfile

from Modules.Modules_Linearized import XiaoiceSing2
from Modules.Discriminator import Discriminator, R1_Regulator
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Noam_Scheduler import Noam_Scheduler
from Logger import Logger

from meldataset import spectral_de_normalize_torch
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.gpu_id == 0:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }
            
            if self.hp.Weights_and_Biases.Use:
                wandb.init(
                    project= self.hp.Weights_and_Biases.Project,
                    entity= self.hp.Weights_and_Biases.Entity,
                    name= self.hp.Weights_and_Biases.Name,
                    config= To_Non_Recursive_Dict(self.hp)
                    )
                wandb.watch(self.model_dict['XiaoiceSing2'])

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        singer_info_dict = yaml.load(open(self.hp.Singer_Info_Path), Loader=yaml.Loader)
        genre_info_dict = yaml.load(open(self.hp.Genre_Info_Path), Loader=yaml.Loader)

        train_dataset = Dataset(
            token_dict= token_dict,
            singer_info_dict= singer_info_dict,
            genre_info_dict= genre_info_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            pattern_length= self.hp.Train.Pattern_Length,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio
            )
        eval_dataset = Dataset(
            token_dict= token_dict,
            singer_info_dict= singer_info_dict,
            genre_info_dict= genre_info_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            pattern_length= self.hp.Train.Pattern_Length,
            accumulated_dataset_epoch= self.hp.Train.Eval_Pattern.Accumulated_Dataset_Epoch,
            )
        inference_dataset = Inference_Dataset(
            token_dict= token_dict,
            singer_info_dict= singer_info_dict,
            genre_info_dict= genre_info_dict,
            durations= self.hp.Train.Inference_in_Train.Duration,
            lyrics= self.hp.Train.Inference_in_Train.Lyric,
            notes= self.hp.Train.Inference_in_Train.Note,
            singers= self.hp.Train.Inference_in_Train.Singer,
            genres= self.hp.Train.Inference_in_Train.Genre,
            sample_rate= self.hp.Sound.Sample_Rate,
            frame_shift= self.hp.Sound.Frame_Shift,
            equality_duration= self.hp.Duration.Equality,
            consonant_duration= self.hp.Duration.Consonant_Duration
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater(
            token_dict= token_dict,
            pattern_length= self.hp.Train.Pattern_Length
            )
        inference_collater = Inference_Collater(
            token_dict= token_dict
            )

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.DistributedSampler(eval_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_dict = {
            'XiaoiceSing2': XiaoiceSing2(self.hp).to(self.device),
            'Discriminator': Discriminator(self.hp).to(self.device),
            }
        self.criterion_dict = {
            'MSE': torch.nn.MSELoss().to(self.device),
            'MAE': torch.nn.L1Loss(reduce= None).to(self.device),
            'R1': R1_Regulator().to(self.device)
            }

        self.optimizer_dict = {
            'XiaoiceSing2': torch.optim.NAdam(
                params= self.model_dict['XiaoiceSing2'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon,
                weight_decay= self.hp.Train.Weight_Decay
                ),
            'Discriminator': torch.optim.NAdam(
                params= self.model_dict['Discriminator'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon,
                weight_decay= self.hp.Train.Weight_Decay
                ),
            }
        self.scheduler_dict = {
            'XiaoiceSing2': Noam_Scheduler(
                optimizer= self.optimizer_dict['XiaoiceSing2'],
                warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step,
                ),
            'Discriminator': Noam_Scheduler(
                optimizer= self.optimizer_dict['Discriminator'],
                warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step,
                ),
            }

        if self.hp.Feature_Type == 'Mel':
            self.vocoder = torch.jit.load('vocgan_sing_mzf_22k_500.pts', map_location='cpu').to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        if self.gpu_id == 0:
            logging.info(self.model_dict)

    def Train_Step(self, tokens, notes, durations, encoding_lengths, singers, genres, features, log_f0s, voices):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)
        encoding_lengths = encoding_lengths.to(self.device, non_blocking=True)
        genres = genres.to(self.device, non_blocking=True)
        singers = singers.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)
        log_f0s = log_f0s.to(self.device, non_blocking=True)
        voices = voices.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            predictions, log_f0_predictions, voice_predictions = self.model_dict['XiaoiceSing2'](
                tokens= tokens,
                notes= notes,
                durations= durations,
                encoding_lengths= encoding_lengths,
                genres= genres,
                singers= singers
                )

            if self.steps >= self.hp.Train.Discrimination_Step:
                features.requires_grad_()
                fake_segment_discriminations_list, _, fake_detail_discriminations_list, _ = self.model_dict['Discriminator'](
                    features= predictions.detach()
                    )
                real_segment_discriminations_list, _, real_detail_discriminations_list, _ = self.model_dict['Discriminator'](
                    features= features
                    )

        if self.steps >= self.hp.Train.Discrimination_Step:
            loss_dict['Discrimination'] = \
                torch.stack([
                    self.criterion_dict['MSE'](discriminations, torch.zeros_like(discriminations)).mean()
                    for discriminations in fake_segment_discriminations_list + fake_detail_discriminations_list
                    ]).mean() + \
                torch.stack([
                    self.criterion_dict['MSE'](discriminations, torch.ones_like(discriminations)).mean()
                    for discriminations in real_segment_discriminations_list + real_detail_discriminations_list
                    ]).mean()
            loss_dict['R1'] = self.criterion_dict['R1'](
                segment_discriminations_list = real_segment_discriminations_list,
                detail_discriminations_list = real_detail_discriminations_list,
                features = features
                )

            self.optimizer_dict['Discriminator'].zero_grad()
            self.scaler.scale(loss_dict['Discrimination'] + loss_dict['R1']).backward()

            if self.hp.Train.Gradient_Norm > 0.0:
                self.scaler.unscale_(self.optimizer_dict['Discriminator'])
                torch.nn.utils.clip_grad_norm_(
                    parameters= self.model_dict['Discriminator'].parameters(),
                    max_norm= self.hp.Train.Gradient_Norm
                    )

            self.scaler.step(self.optimizer_dict['Discriminator'])
            self.scaler.update()
            self.scheduler_dict['Discriminator'].step()

        loss_dict['Feature'] = self.criterion_dict['MSE'](predictions, features)
        loss_dict['Log_F0'] = self.criterion_dict['MSE'](log_f0_predictions, log_f0s)
        loss_dict['Voice'] = self.criterion_dict['MSE'](voice_predictions, voices)
        loss = loss_dict['Feature'] + loss_dict['Log_F0'] + loss_dict['Voice']

        if self.steps >= self.hp.Train.Discrimination_Step:
            with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
                fake_segment_discriminations_list, fake_segment_feature_maps_list, \
                fake_detail_discriminations_list, fake_detail_feature_maps_list = self.model_dict['Discriminator'](
                    features= predictions
                    )
                _, real_segment_feature_maps_list, \
                _, real_detail_feature_maps_list = self.model_dict['Discriminator'](
                    features= features
                    )
            
            loss_dict['Adversarial'] = torch.stack([
                self.criterion_dict['MSE'](discriminations, torch.ones_like(discriminations)).mean()
                for discriminations in fake_segment_discriminations_list + fake_detail_discriminations_list
                ]).mean()
            feature_matching_loss = torch.stack([
                self.criterion_dict['MAE'](fake_discrimination_features, real_discrimination_features).mean()
                for fake_discrimination_features, real_discrimination_features in zip(
                    fake_segment_feature_maps_list + fake_detail_feature_maps_list,
                    real_segment_feature_maps_list + real_detail_feature_maps_list
                    )
                ]).mean()
            loss = loss + feature_matching_loss * loss_dict['Adversarial']

        self.optimizer_dict['XiaoiceSing2'].zero_grad()
        self.scaler.scale(loss).backward()
        if self.hp.Train.Gradient_Norm > 0.0:
            self.scaler.unscale_(self.optimizer_dict['XiaoiceSing2'])
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_dict['XiaoiceSing2'].parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer_dict['XiaoiceSing2'])
        self.scaler.update()
        self.scheduler_dict['XiaoiceSing2'].step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for tokens, notes, durations, encoding_lengths, singers, genres, features, log_f0s, voices in self.dataloader_dict['Train']:
            self.Train_Step(tokens, notes, durations, encoding_lengths, singers, genres, features, log_f0s, voices)

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler_dict['XiaoiceSing2'].get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            f'Train.{key}': value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return


    def Evaluation_Step(self, tokens, notes, durations, encoding_lengths, singers, genres, features, log_f0s, voices):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)
        encoding_lengths = encoding_lengths.to(self.device, non_blocking=True)
        genres = genres.to(self.device, non_blocking=True)
        singers = singers.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)
        log_f0s = log_f0s.to(self.device, non_blocking=True)
        voices = voices.to(self.device, non_blocking=True)

        predictions, log_f0_predictions, voice_predictions = self.model_dict['XiaoiceSing2'](
            tokens= tokens,
            notes= notes,
            durations= durations,
            encoding_lengths= encoding_lengths,
            genres= genres,
            singers= singers
            )

        loss_dict['Feature'] = self.criterion_dict['MSE'](predictions, features)
        loss_dict['Log_F0'] = self.criterion_dict['MSE'](log_f0_predictions, log_f0s)
        loss_dict['Voice'] = self.criterion_dict['MSE'](voice_predictions, voices)

        if self.steps >= self.hp.Train.Discrimination_Step:
            features.requires_grad_()
            fake_segment_discriminations_list, fake_segment_feature_maps_list, \
            fake_detail_discriminations_list, fake_detail_feature_maps_list = self.model_dict['Discriminator'](
                features= predictions
                )
            real_segment_discriminations_list, real_segment_feature_maps_list, \
            real_detail_discriminations_list, real_detail_feature_maps_list = self.model_dict['Discriminator'](
                features= features
                )

            loss_dict['Discrimination'] = \
                torch.stack([
                    self.criterion_dict['MSE'](discriminations, torch.zeros_like(discriminations)).mean()
                    for discriminations in fake_segment_discriminations_list + fake_detail_discriminations_list
                    ]).mean() + \
                torch.stack([
                    self.criterion_dict['MSE'](discriminations, torch.ones_like(discriminations)).mean()
                    for discriminations in real_segment_discriminations_list + real_detail_discriminations_list
                    ]).mean()
            loss_dict['Adversarial'] = torch.stack([
                self.criterion_dict['MSE'](discriminations, torch.ones_like(discriminations)).mean()
                for discriminations in fake_segment_discriminations_list + fake_detail_discriminations_list
                ]).mean()
            loss_dict['R1'] = self.criterion_dict['R1'](
                segment_discriminations_list = real_segment_discriminations_list,
                detail_discriminations_list = real_detail_discriminations_list,
                features = features
                )

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

        return predictions, log_f0_predictions, voice_predictions

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        for model in self.model_dict.values():
            model.eval()

        for step, (tokens, notes, durations, encoding_lengths, singers, genres, features, log_f0s, voices) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            predictions, log_f0_predictions, voice_predictions = \
                self.Evaluation_Step(tokens, notes, durations, encoding_lengths, singers, genres, features, log_f0s, voices)
        
        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['XiaoiceSing2'], 'XiaoiceSing2', self.steps, delete_keywords=[])
            self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['Discriminator'], 'Discriminator', self.steps, delete_keywords=[])
        
            index = np.random.randint(0, tokens.size(0))
            feature_length = durations.sum(dim= 1)[index].cpu().numpy()

            target_feature = features[index, :, :feature_length]
            prediction_feature = predictions.detach()[index, :, :feature_length]

            log_f0_target = log_f0s[index, :feature_length].cpu().numpy()
            log_f0_prediction = log_f0_predictions.detach()[0, :feature_length].cpu().numpy()

            if self.hp.Feature_Type == 'Mel':
                target_audio = self.vocoder(target_feature.unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy() / 32768.0
                prediction_audio = self.vocoder(prediction_feature.unsqueeze(0)).squeeze(0).cpu().numpy() / 32768.0
            elif self.hp.Feature_Type == 'Spectrogram':
                target_audio = griffinlim(spectral_de_normalize_torch(target_feature.squeeze(0)).cpu().numpy())
                prediction_audio = griffinlim(spectral_de_normalize_torch(prediction_feature.squeeze(0)).cpu().numpy())

            image_dict = {
                'Feature/Target': (target_feature.squeeze(0).cpu().numpy(), None, 'auto', None, None, None),
                'Feature/Prediction': (prediction_feature.squeeze(0).cpu().numpy(), None, 'auto', None, None, None),
                'Log_F0/Target': (log_f0_target, None, 'auto', None, None, None),
                'Log_F0/Prediction': (log_f0_prediction, None, 'auto', None, None, None)
                }
            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)

            audio_dict = {
                'Audio/Target': (target_audio, self.hp.Sound.Sample_Rate),
                'Audio/Prediction': (prediction_audio, self.hp.Sound.Sample_Rate),
                }
            self.writer_dict['Evaluation'].add_audio_dict(audio_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        f'Evaluation.{key}': value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                wandb.log(
                    data= {                        
                        'Evaluation.Feature.Target': wandb.Image(target_feature.squeeze(0).cpu().numpy()),
                        'Evaluation.Feature.Linear': wandb.Image(prediction_feature.squeeze(0).cpu().numpy()),
                        'Evaluation.Log_F0': wandb.plot.line_series(
                            xs= np.arange(feature_length),
                            ys= [log_f0_target, log_f0_prediction],
                            keys= ['Target', 'Prediction'],
                            title= 'Log_F0',
                            xname= 'Token_t'
                            ),
                        'Evaluation.Audio.Target': wandb.Audio(
                            target_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Prediction'
                            ),
                        'Evaluation.Audio.Linear': wandb.Audio(
                            prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Prediction'
                            ),
                        },
                    step= self.steps,
                    commit= False
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)

        for model in self.model_dict.values():
            model.train()


    @torch.inference_mode()
    def Inference_Step(self, tokens, notes, durations, encoding_lengths, singers, genres, lyrics, start_index= 0, tag_step= False):
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)
        encoding_lengths = encoding_lengths.to(self.device, non_blocking=True)
        genres = genres.to(self.device, non_blocking=True)
        singers = singers.to(self.device, non_blocking=True)

        predictions, log_f0_predictions, voice_predictions = self.model_dict['XiaoiceSing2'](
            tokens= tokens,
            notes= notes,
            durations= durations,
            encoding_lengths= encoding_lengths,
            genres= genres,
            singers= singers
            )
        
        if self.hp.Feature_Type == 'Mel':
            audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))].cpu().numpy()
                for audio, length in zip(
                    self.vocoder(predictions),
                    durations.sum(dim= 1)
                    )
                ]
        elif self.hp.Feature_Type == 'Spectrogram':
            audios = []
            for prediction, length in zip(
                predictions,
                durations.sum(dim= 1)
                ):
                prediction = spectral_de_normalize_torch(prediction).cpu().numpy()
                audio = griffinlim(prediction)[:min(prediction.size(1), length) * self.hp.Sound.Frame_Shift]
                audio = (audio / np.abs(audio).max() * 32767.5).astype(np.int16)
                audios.append(audio)
                
        files = []
        for index in range(predictions.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        durations = [
            torch.arange(duration.size(0)).repeat_interleave(duration.cpu()).numpy()
            for duration in durations
            ]

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (prediction, duration, log_f0, encoding_length, feature_length, lyric, audio, file) in enumerate(zip(
            predictions.cpu().numpy(),
            durations,
            log_f0_predictions.cpu().numpy(),
            encoding_lengths.cpu().numpy(),
            [duration[:encoding_length].sum() for duration, encoding_length in zip(durations, encoding_lengths)],
            lyrics,
            audios,
            files
            )):
            title = 'Lyric: {}'.format(lyric if len(lyric) < 90 else lyric[:90] + '…')
            new_figure = plt.figure(figsize=(20, 5 * 4), dpi=100)
            ax = plt.subplot2grid((4, 1), (0, 0))
            plt.imshow(prediction[:, :feature_length], aspect='auto', origin='lower')
            plt.title('Prediction    {}'.format(title))
            plt.colorbar(ax= ax)            
            ax = plt.subplot2grid((4, 1), (1, 0), rowspan= 2)
            plt.plot(duration[:feature_length])
            plt.title('Duration    {}'.format(title))
            plt.margins(x= 0)            
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((4, 1), (3, 0))
            plt.plot(log_f0[:feature_length])
            plt.title('Log F0    {}'.format(title))
            plt.margins(x= 0)            
            plt.colorbar(ax= ax)
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_figure)

            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                audio
                )
            
    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return
            
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_dict.values():
            model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (tokens, notes, durations, encoding_lengths, singers, genres, lyrics) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(tokens, notes, durations, encoding_lengths, singers, genres, lyrics, start_index= step * batch_size)

        for model in self.model_dict.values():
            model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_dict = torch.load(path, map_location= 'cpu')
        self.model_dict['XiaoiceSing2'].load_state_dict(state_dict['Model']['XiaoiceSing2'])
        self.model_dict['Discriminator'].load_state_dict(state_dict['Model']['Discriminator'])
        self.optimizer_dict['XiaoiceSing2'].load_state_dict(state_dict['Optimizer']['XiaoiceSing2'])
        self.optimizer_dict['Discriminator'].load_state_dict(state_dict['Optimizer']['Discriminator'])
        self.scheduler_dict['XiaoiceSing2'].load_state_dict(state_dict['Scheduler']['XiaoiceSing2'])
        self.scheduler_dict['Discriminator'].load_state_dict(state_dict['Scheduler']['Discriminator'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_dict = {
            'Model': {
                'XiaoiceSing2': self.model_dict['XiaoiceSing2'].state_dict(),
                'Discriminator': self.model_dict['Discriminator'].state_dict(),
                },
            'Optimizer': {
                'XiaoiceSing2': self.optimizer_dict['XiaoiceSing2'].state_dict(),
                'Discriminator': self.optimizer_dict['Discriminator'].state_dict(),
                },
            'Scheduler': {
                'XiaoiceSing2': self.scheduler_dict['XiaoiceSing2'].state_dict(),
                'Discriminator': self.scheduler_dict['Discriminator'].state_dict(),
                },
            'Steps': self.steps
            }
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        torch.save(state_dict, checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)


    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model_dict = apply_gradient_allreduce(self.model_dict)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_path, hp_path)

        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)    
    argParser.add_argument('-p', '--port', default= 54321, type= int)
    argParser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl'
            )
    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()