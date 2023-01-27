import torch
import numpy as np
import logging, yaml, os, sys, argparse, math
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple
from librosa import griffinlim
from scipy.io import wavfile

from Modules.Modules_Linearized import XiaoiceSing2
from Datasets import Inference_Dataset as Dataset, Inference_Collater as Collater
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
            # self.vocoder = torch.jit.load('vocgan_sing_mzf_22k_500.pts', map_location='cpu').to(self.device)
            self.vocoder = torch.jit.load('universal_0250.pts', map_location='cpu').to(self.device)

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
    def Inference_Step(self, tokens, notes, durations, encoding_lengths, singers, genres, lyrics):
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)
        encoding_lengths = encoding_lengths.to(self.device, non_blocking=True)
        singers = singers.to(self.device, non_blocking=True)
        genres = genres.to(self.device, non_blocking=True)
        
        predictions, log_f0_predictions, voice_predictions = self.model(
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
        for tokens, notes, durations, encoding_lengths, singers, genres, lyrics in dataloader:
            audios.extend(self.Inference_Step(tokens, notes, durations, encoding_lengths, singers, genres, lyrics))
            
        
        return audios

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-checkpoint', '--checkpoint', type= str, required= True)
    parser.add_argument('-batch', '--batch', default= 1, type= int)
    args = parser.parse_args()
    
    inferencer = Inferencer(
        hp_path= args.hyper_parameters,
        checkpoint_path= args.checkpoint,
        batch_size= args.batch
        )

    patterns = []
    for path in [
        # './Inference_for_Training/Example1.txt',
        # './Inference_for_Training/Example2.txt',
        # './Inference_for_Training/Example3.txt',
        './Inference_for_Training/Example4.txt',
        './Inference_for_Training/Example5.txt',
        ]:        
        pattern = []
        for line in open(path, 'r', encoding= 'utf-8').readlines()[1:]:
            duration, text, note = line.strip().split('\t')
            pattern.append((int(duration), text, int(note)))
        patterns.append(pattern)
    audios = inferencer.Inference_Epoch(patterns, True)

# python Inference.py -hp Hyper_Parameters.yaml -checkpoint /data/results/MLPSinger/MLPSinger.Spect/Checkpoint/S_100000.pt -outdir ./results/