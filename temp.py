# # import pickle, yaml, os
# # from tqdm import tqdm
# # from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict

# # hp = Recursive_Parse(yaml.load(
# #     open('Hyper_Parameters.yaml', encoding='utf-8'),
# #     Loader=yaml.Loader
# #     ))

# # metadata_dict = pickle.load(open(
# #     os.path.join(hp.Train.Eval_Pattern.Path, hp.Train.Eval_Pattern.Metadata_File).replace('\\', '/'), 'rb'
# #     ))

# # for patterns in tqdm(metadata_dict['File_List']):
# #     path = os.path.join(hp.Train.Eval_Pattern.Path, patterns).replace('\\', '/')
# #     pattern_dict = pickle.load(open(path, 'rb'))
# #     if pattern_dict['Genre'] == 'Pop':
# #         pattern_dict['Genre'] = 'Trot'
# #         pickle.dump(
# #             pattern_dict,
# #             open(path, 'wb'),
# #             protocol= 4
# #             )

# # import torch

# # layer = torch.nn.TransformerEncoderLayer(512, 8)
# # net = torch.nn.TransformerEncoder(layer, 6)
# # net.layers[0].linear1.weight == net.layers[1].linear1.weight



# # import mido

# # mid = mido.MidiFile('C:/Users/Heejo.You/Desktop/07697.mid', charset='CP949')
# # music = []
# # current_lyric = ''
# # current_note = None
# # current_time = 0.0

# # # Note on 쉼표
# # # From Lyric to message before note on: real note
# # for message in list(mid):
# #     if message.type == 'note_on' and message.velocity != 0:
# #         if message.time < 0.1:
# #             current_time += message.time
# #             if current_lyric in ['J', 'H', None]:
# #                 music.append((current_time, '<X>', 0))
# #             else:
# #                 music.append((current_time, current_lyric, current_note))
# #         else:
# #             if not current_lyric in ['J', 'H', None]:
# #                 music.append((current_time, current_lyric, current_note))
# #             else:
# #                 message.time += current_time
# #             music.append((message.time, '<X>', 0))
# #         current_time = 0.0
# #         current_lyric = ''
# #         current_note = None
# #     elif message.type == 'lyrics':
# #         if message.text == '\r':    # mzm 02678.mid
# #             continue
# #         current_lyric = message.text.strip()
# #         current_time += message.time
# #     elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
# #         current_note = message.note
# #         current_time += message.time
# #         if current_lyric == 'H':    # it is temp.
# #             break
# #     else:
# #         current_time += message.time

# # if current_lyric in ['J', 'H']:
# #     if music[-1][1] == '<X>':
# #         music[-1] = (music[-1][0] + current_time, music[-1][1], music[-1][2])
# #     else:
# #         music.append((current_time, '<X>', 0))
# # else:
# #     music.append((current_time, current_lyric, current_note))
# # music = music[1:]

# # print(music)

# # import pickle, yaml, os
# # from tqdm import tqdm
# # from scipy.io import wavfile
# # from Inference import Inferencer
# # from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict

# # hp = Recursive_Parse(yaml.load(
# #     open('Hyper_Parameters.yaml', encoding='utf-8'),
# #     Loader=yaml.Loader
# #     ))

# # inferencer = Inferencer(
# #     hp_path= 'Hyper_Parameters.yaml',
# #     checkpoint_path= 'C:/Users/Heejo.You/Downloads/Telegram Desktop/S_200000_Five_Correction.pt',
# #     overlapped_frame= 16,
# #     batch_size= 16
# #     )

# # message_times, lyrics, notes = zip(*music)
# # message_times = [message_times] * 4
# # lyrics = [lyrics] * 4
# # notes = [notes] * 4
# # singers = [
# #     'Mediazen_Female',
# #     'Mediazen_Male',
# #     'KJE',
# #     'NAMS',
# #     ]
# # genres = [
# #     'Ballade',
# #     'Ballade',
# #     'Ballade',
# #     'Ballade',    
# #     ]

# # audios = inferencer.Inference_Epoch(
# #     message_times_list= message_times,
# #     lyrics= lyrics,
# #     notes= notes,
# #     singers= singers,
# #     genres= genres
# #     )

# # for index, audio in enumerate(audios):
# #     wavfile.write(f'{index}_Octave_Correction.wav', 22050, audio)

# import torch

# timesteps = 100
# betas = torch.linspace(1e-4, 0.06, timesteps)
# alphas = 1.0 - betas
# alphas_cumprod = torch.cumprod(alphas, axis= 0)
# alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
# sqrt_alphas_cumprod = alphas_cumprod.sqrt()
# sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
# sqrt_recip_alphas_cumprod = (1.0 / alphas_cumprod).sqrt()
# sqrt_recipm1_alphas_cumprod = (1.0 / alphas_cumprod - 1.0).sqrt()


# diffusion_steps = 10
# x1 = sqrt_alphas_cumprod[10]
# x2 = 1e-4 * diffusion_steps + 0.5 * (0.6 - 1e-4) * diffusion_steps ** 2

# import pickle, yaml
# from scipy.io import wavfile
# from Inference import Inferencer, Collater, Lyric_to_Token

# inferencer = Inferencer(
#     hp_path= 'Hyper_Parameters.yaml',
#     checkpoint_path= 'C:/Users/Heejo.You/Downloads/Telegram Desktop/S_200000_UNet2_Yua_Duration_Fix.pt',
#     # checkpoint_path= 'C:/Users/Heejo.You/Downloads/Telegram Desktop/S_500000.pt',
#     overlapped_frame= 16,
#     batch_size= 16
#     )

# token_dict = yaml.load(open(inferencer.hp.Token_Path), Loader=yaml.Loader)
# singer_info_dict = yaml.load(open(inferencer.hp.Singer_Info_Path), Loader=yaml.Loader)
# genre_info_dict = yaml.load(open(inferencer.hp.Genre_Info_Path), Loader=yaml.Loader)

# pattern = pickle.load(open('E:/22K.Music/Eval/CSD/CSD/kr050b.pickle', 'rb'))

# collater = Collater(
#     token_dict= token_dict,
#     pattern_length= inferencer.hp.Train.Pattern_Length,
#     overlapped_frame= 128
#     )

# tokens, notes, singers, genres, lengths, lyrics = collater([(
#     Lyric_to_Token(pattern['Lyric'], token_dict),
#     [x + 12 for x in pattern['Note']],
#     singer_info_dict['KJE'],
#     genre_info_dict[pattern['Genre']],
#     ''
#     )])

# audios = inferencer.Inference_Step(
#     tokens= tokens,
#     notes= notes,
#     singers= singers,
#     genres= genres,
#     lengths= lengths,
#     lyrics= lyrics
#     )

# wavfile.write('KJE_200K.wav', 22050, audios[0])

# import pickle

# scores = open('/datasets/rawdata_music/CSD/korean/csv/kr013b.csv', encoding='utf-8-sig').readlines()[1:]
# lyrics = open('/datasets/rawdata_music/CSD/korean/lyric/kr013b.txt', encoding='utf-8-sig').read().strip().replace(' ', '').replace('\n', '')

# music = []
# previous_end_time = 0.0
# for score, lyric in zip(scores, lyrics):
#     start_time, end_time, note, _, = score.strip().split(',')
#     start_time, end_time, note = float(start_time), float(end_time), int(note)

#     if start_time != previous_end_time:
#         music.append((start_time - previous_end_time, '<X>', 0))
#     music.append((float(f'{end_time - start_time:.3f}'), lyric, note))
#     previous_end_time = end_time

# pickle.dump(music, open('kr013b.pickle', 'wb'))

# import pickle
# import mido
# import os

# midi_path = '/datasets/rawdata_music/Nams/00444.mid'
# mid = mido.MidiFile(midi_path, charset='CP949')
# music = []
# current_lyric = ''
# current_note = None
# current_time = 0.0
# # Note on 쉼표
# # From Lyric to message before note on: real note
# for message in list(mid):
#     if message.type == 'note_on' and message.velocity != 0:
#         if message.time < 0.1:
#             current_time += message.time
#             if current_lyric in ['J', 'H', None]:
#                 music.append((current_time, '<X>', 0))
#             else:
#                 music.append((current_time, current_lyric, current_note))
#         else:
#             if not current_lyric in ['J', 'H', None]:
#                 music.append((current_time, current_lyric, current_note))
#             else:
#                 message.time += current_time
#             music.append((message.time, '<X>', 0))
#         current_time = 0.0
#         current_lyric = ''
#         current_note = None
#     elif message.type == 'lyrics':
#         if message.text == '\r':    # mzm 02678.mid
#             continue
#         current_lyric = message.text.strip()
#         current_time += message.time
#     elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
#         current_note = message.note
#         current_time += message.time
#         if current_lyric == 'H':    # it is temp.
#             break
#     else:
#         current_time += message.time

# if current_lyric in ['J', 'H']:
#     if music[-1][1] == '<X>':
#         music[-1] = (music[-1][0] + current_time, music[-1][1], music[-1][2])
#     else:
#         music.append((current_time, '<X>', 0))
# else:
#     music.append((current_time, current_lyric, current_note))
# music = music[1:]
# pickle.dump(music, open('00444.pickle', 'wb'))


# from Inference import Inferencer
# from scipy.io import wavfile
# import os

# durations = [
#     [0.326,0.163,0.326,0.489,0.326,0.163,0.814,0.327,0.163,0.163,0.326,0.163,0.489,0.163,0.815,0.326,0.163,0.326,0.489,0.326,0.163,0.326,0.489,0.326,0.326,0.163,0.326,1.467,0.326,0.163,0.326,0.489,0.326,0.163,0.814,0.327,0.163,0.163,0.326,0.163,0.489,0.163,0.815,0.326,0.163,0.326,0.163,0.326,0.489,0.163,0.326,0.326,0.326,0.326,0.163,0.326,0.815],
#     ] * 4
# lyrics = [
#     ['마','음','울','적','한','날','에','<X>','거','리','를','걸','어','보','고','향','기','로','운','칵','테','일','에','취','해','도','보','고','한','편','의','시','가','있','는','<X>','전','시','회','장','도','가','고','밤','새','도','<X>','록','그','리','움','에','편','질','쓰','고','파',],
#     ] * 4
# notes = [
#     [68,68,68,75,73,72,70,0,72,72,72,73,72,67,67,65,65,65,68,68,66,65,63,65,68,67,68,70,68,68,68,75,73,72,70,0,72,72,72,73,72,67,67,65,65,65,67,68,68,65,63,63,65,68,67,70,68,],
#     ] * 4
# singers = [
#     'Mediazen_Female',
#     'Mediazen_Male',
#     'KJE',
#     'TTS_SGHVC_Yura',
#     ]
# genres = [
#     'Anime',
#     'Anime',
#     'Anime',
#     'Anime',
#     ]
# notes = [[(x + 12) if x != 0 else 0 for x in xx] for xx in notes]

# os.makedirs('inference_test', exist_ok= True)

# for model_index, checkpoint_path in enumerate([
#     '/data/results/PTransDiffSVS/TTS_Yura_0/Checkpoint/S_1000000.pt',
#     '/data/results/PTransDiffSVS/TTS_Yura_1/Checkpoint/S_200000.pt',
#     '/data/results/PTransDiffSVS/TTS_Yura_2/Checkpoint/S_200000.pt',
#     '/data/results/PTransDiffSVS/TTS_Yura_3/Checkpoint/S_200000.pt',
#     '/data/results/PTransDiffSVS/TTS_Yura_4/Checkpoint/S_200000.pt',
#     '/data/results/PTransDiffSVS/TTS_Yura_5/Checkpoint/S_200000.pt',
#     '/data/results/PTransDiffSVS/TTS_Yura_6/Checkpoint/S_200000.pt',
#     '/data/results/PTransDiffSVS/TTS_Yura_7/Checkpoint/S_200000.pt',
#     '/data/results/PTransDiffSVS/TTS_Yura_8/Checkpoint/S_200000.pt',
#     '/data/results/PTransDiffSVS/TTS_Yura_9/Checkpoint/S_200000.pt',
#     ]):
#     inferencer = Inferencer(
#         hp_path= 'Hyper_Parameters.yaml',
#         checkpoint_path= checkpoint_path,
#         batch_size= 16
#         )
#     _, diffusion_audios = inferencer.Inference_Epoch(
#         message_times_list= durations,
#         lyrics= lyrics,
#         notes= notes,
#         singers= singers,
#         genres= genres
#         )
#     for audio, singer in zip(diffusion_audios, singers):
#         wavfile.write(
#             f'inference_test/Model_{model_index}.Singer_{singer}.wav',
#             inferencer.hp.Sound.Sample_Rate,
#             audio
#             )

# import mido
# import os

# for root, _, files in os.walk('/datasets/rawdata_music/Fix_Test/Nams/mid'):
#     for file in files:        
#         mid = mido.MidiFile(os.path.join(root, file), charset='cp949')
#         track = sorted([(len(x), x) for x in mid.tracks], key= lambda x: x[0], reverse= True)[0][1]

#         is_press = False
#         for index, message in enumerate(track):
#             if not message.type in ['note_on', 'note_off', 'lyrics']:
#                 continue
#             if message.type == 'lyrics' and not is_press:
#                 # print(file, index)
#                 # for i, x in enumerate(track[max(0, index-5):index+5]):
#                 #     print(i+max(0, index-5), x)
#                 # print('#' * 50)
#                 # break
#                 print(file, f'{index / len(track):.3f}', ''.join([x.text for x in track[index:index+20] if x.type == 'lyrics']), sep='\t')
#                 # print('#' * 50)
#                 break
#                 # print(file)
#                 # break
#             if message.type in ['note_on', 'note_off'] and message.velocity > 0:
#                 is_press = True
#             if message.type in ['note_on', 'note_off'] and message.velocity == 0:
#                 is_press = False

        
# import torch
# from Modules.Modules import Positional_Encoding

# pe = Positional_Encoding(384)

# x = torch.randn(16, 384, 963)
# y = pe(x)

import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pysptk.sptk import rapt
from meldataset import mel_spectrogram
import parselmouth

path = 'D:/Datasets/LJSpeech/wavs/LJ001-0010.wav'

audio, sample_rate = librosa.load(path)
pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
    audio,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7'),
    frame_length=1024
    )
pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)


f0 = rapt(
    x= (audio * 32768).astype(np.int16),
    fs= sample_rate,
    hopsize= 256,
    min= librosa.note_to_hz('C2'),
    max= librosa.note_to_hz('C7'),
    otype= 1
    )

mel = mel_spectrogram(
    y= torch.from_numpy(audio).float().unsqueeze(0),
    n_fft= 1024,
    num_mels= 80,
    sampling_rate= sample_rate,
    hop_size= 256,
    win_size= 1024,
    fmin= librosa.note_to_hz('C2'),
    fmax= librosa.note_to_hz('C7'),
    center= False
    ).squeeze(0).numpy()


audio = parselmouth.Sound(path)
parselmouth_pitch = audio.to_pitch().selected_array['frequency']

plt.subplot(511)
plt.plot(audio)
plt.margins(x= 0)
plt.subplot(512)
plt.imshow(mel, aspect='auto', origin= 'lower')
plt.subplot(513)
plt.plot(pitch_mel)
plt.margins(x= 0)
plt.subplot(514)
plt.plot(f0)
plt.margins(x= 0)
plt.subplot(515)
plt.plot(parselmouth_pitch)
plt.margins(x= 0)
plt.show()

print()
