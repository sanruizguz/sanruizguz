import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import maad
from maad import sound
from scipy import signal
from maad import util
fmin = 2000
fmax = 6000
path_of_the_directory="J:/2023/Wav_divididos/Originals/E"

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

list_nparray= []
list_png= []

for recording in glob.iglob(f'{path_of_the_directory}/*.wav'): 
    file_name = os.path.basename(recording)
    name = (os.path.splitext(file_name)[0])
    name = name + '.png'    
    fig, ax = plt.subplots()
    s, fs = sound.load(recording)

    s_trim = maad.sound.trim(s,fs, 0,3,pad=True,pad_constant=0)
    D_highres = librosa.stft(s_trim)
    S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)

    plt.ylim([1000, 4500])
    img = librosa.display.specshow(S_db_hr
                               ,cmap='gray_r'    
                               ,hop_length=156
                               , x_axis='time'
                               , y_axis='linear'
                               ,ax=ax)
    list_png.append([img])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(fname=name)
    plt.close()
