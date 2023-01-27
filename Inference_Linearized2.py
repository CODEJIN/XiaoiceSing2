import torch
import numpy as np
import logging, yaml, os, sys, argparse, math
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import griffinlim

from Modules.Modules_Linearized2 import XiaoiceSing2
from Datasets2 import Inference_Dataset as Dataset, Inference_Collater as Collater
from meldataset import spectral_de_normalize_torch
from Arg_Parser import Recursive_Parse

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )


class Inferencer:
    def __init__(
        self,
        hp_path: str,
        checkpoint_path: str,
        batch_size= 1
        ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.model = XiaoiceSing2(self.hp).to(self.device)
        if self.hp.Feature_Type == 'Mel':
            self.vocoder = torch.jit.load('hifigan_jit_sing_0273.pts', map_location='cpu').to(self.device)
            # self.vocoder = torch.jit.load('universal_0250.pts', map_location='cpu').to(self.device)

        if self.hp.Feature_Type == 'Spectrogram':
            self.feature_range_info_dict = yaml.load(open(self.hp.Spectrogram_Range_Info_Path), Loader=yaml.Loader)
        if self.hp.Feature_Type == 'Mel':
            self.feature_range_info_dict = yaml.load(open(self.hp.Mel_Range_Info_Path), Loader=yaml.Loader)
        self.index_singer_dict = {
            value: key
            for key, value in yaml.load(open(self.hp.Singer_Info_Path), Loader=yaml.Loader).items()
            }

        if self.hp.Feature_Type == 'Spectrogram':
            self.feature_size = self.hp.Sound.N_FFT // 2 + 1
        elif self.hp.Feature_Type == 'Mel':
            self.feature_size = self.hp.Sound.Mel_Dim
        else:
            raise ValueError('Unknown feature type: {}'.format(self.hp.Feature_Type))

        self.Load_Checkpoint(checkpoint_path)
        self.batch_size = batch_size

    def Dataset_Generate(self, message_times_list, lyrics, notes, singers, genres):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        singer_info_dict = yaml.load(open(self.hp.Singer_Info_Path), Loader=yaml.Loader)
        genre_info_dict = yaml.load(open(self.hp.Genre_Info_Path), Loader=yaml.Loader)

        return torch.utils.data.DataLoader(
            dataset= Dataset(
                token_dict= token_dict,
                singer_info_dict= singer_info_dict,
                genre_info_dict= genre_info_dict,
                durations= message_times_list,
                lyrics= lyrics,
                notes= notes,
                singers= singers,
                genres= genres,
                sample_rate= self.hp.Sound.Sample_Rate,
                frame_shift= self.hp.Sound.Frame_Shift,
                equality_duration= self.hp.Duration.Equality,
                consonant_duration= self.hp.Duration.Consonant_Duration
                ),
            shuffle= False,
            collate_fn= Collater(
                token_dict= token_dict
                ),
            batch_size= self.batch_size,
            num_workers= 0,
            pin_memory= True
            )

    def Load_Checkpoint(self, path):
        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model']['XiaoiceSing2'])        
        self.steps = state_dict['Steps']

        self.model.eval()

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    @torch.no_grad()
    def Inference_Step(self, tokens, notes, lengths, singers, genres, lyrics):
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device, non_blocking=True)
        singers = singers.to(self.device, non_blocking=True)
        genres = genres.to(self.device, non_blocking=True)
        
        predictions, log_f0_predictions, voice_predictions = self.model(
            tokens= tokens,
            notes= notes,
            lengths= lengths,
            genres= genres,
            singers= singers
            )
        
        if self.hp.Feature_Type == 'Mel':
            audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))].cpu().numpy()
                for audio, length in zip(
                    self.vocoder(predictions),
                    lengths
                    )
                ]
        elif self.hp.Feature_Type == 'Spectrogram':
            audios = []
            for prediction, length in zip(
                predictions,
                lengths
                ):
                prediction = spectral_de_normalize_torch(prediction).cpu().numpy()
                audio = griffinlim(prediction)[:min(prediction.size(1), length) * self.hp.Sound.Frame_Shift]
                audio = (audio / np.abs(audio).max() * 32767.5).astype(np.int16)
                audios.append(audio)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 10))
        plt.imshow(predictions[0].cpu().numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('x.png')

        return audios

    def Inference_Epoch(self, message_times_list, lyrics, notes, singers, genres, use_tqdm= True):
        dataloader = self.Dataset_Generate(
            message_times_list= message_times_list,
            lyrics= lyrics,
            notes= notes,
            singers= singers,
            genres= genres
            )
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )
        audios = []
        for tokens, notes, lengths, singers, genres, lyrics in dataloader:
            audios.extend(self.Inference_Step(tokens, notes, lengths, singers, genres, lyrics))
            
        
        return audios