import numpy as np
import matplotlib.pyplot as plt
import pickle, hgtk, os
from matplotlib import rc
rc("font", family="Malgun Gothic")

class Note_Predictor:
    def __init__(self):
        notes = np.arange(0, 128)
        # log_f0s = np.log(440 * 2 ** ((notes - 69) / 12))
        log_f0s = np.log(440 * 2 ** ((notes - 69 - 12) / 12))
        log_f0s[0] = 0.0
        self.criterion = np.expand_dims(log_f0s, axis= 0)   # [1, 128]

    def __call__(self, log_f0):
        '''
        log_f0: [F0_t]
        '''
        return np.argmin(
            np.abs(np.expand_dims(log_f0, axis= 1) - self.criterion),
            axis= 1
            )

def Lyric_Compose(lyric):
    composed_lyric = []
    current_string = []
    for letter in lyric + ['Temp_Lyric']:
        if len(current_string) > 0 and current_string[-1] != letter:
            if current_string[-1][-1] != '_' and not current_string[-1] in hgtk.checker.JAMO:
                composed_lyric.append(current_string[0])
                composed_lyric.extend([None] * (len(current_string) - 1))
                current_string = []
            elif current_string[-1][-1] == '_':
                for x in set(current_string):
                    if x in hgtk.letter.CHO:
                        onset = x
                    elif x in hgtk.letter.JOONG:
                        nuclues = x
                    elif len(x) == 2 and x[:1] in hgtk.letter.JONG:
                        coda = x[0]
                    elif x == '_':
                        coda = ''
                    else:
                        raise ValueError(set(current_string))
                composed_lyric.append(hgtk.letter.compose(onset, nuclues, coda))
                composed_lyric.extend([None] * (len(current_string) - 1))
                current_string = []

        current_string.append(letter)

    return composed_lyric

os.makedirs('./SVS_Pattern_Test/', exist_ok= True)
for root, _, files in os.walk('./KJE'):
    for file in files:
        pattern = pickle.load(open(os.path.join(root, file), 'rb'))
        note_predictor = Note_Predictor()

        lyric = Lyric_Compose(pattern['Lyric'])

        plt.figure(figsize= (200, 20))
        plt.subplot(211)
        plt.imshow(pattern['Mel'].T, aspect='auto', origin='lower')
        plt.subplot(212)
        plt.plot(note_predictor(np.clip(pattern['Log_F0'], -0.0, np.inf)))
        plt.plot(pattern['Note'])
        plt.xticks(
            range(len(lyric)),
            lyric,
            fontsize = 8
            )
        plt.margins(x= 0)
        plt.tight_layout()
        plt.savefig(f'./SVS_Pattern_Test/{os.path.splitext(file)[0]}.png')
        plt.close()
