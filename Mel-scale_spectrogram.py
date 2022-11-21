import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import maad
from maad import sound
from scipy import signal

fmin = 2000
fmax = 6000
path_of_the_directory="C:/Users/sruizguz/Downloads/Labeling project recordings-selected"

list_nparray= []
list_png= []

for recording in glob.iglob(f'{path_of_the_directory}/*.wav'): 
    file_name = os.path.basename(recording)
    name = (os.path.splitext(file_name)[0])
    name = name + '.png'
    print(name)
    print(recording)
    
list_nparray= []
list_png= []

for recording in glob.iglob(f'{path_of_the_directory}/*.wav'): 
    file_name = os.path.basename(recording)
    name = (os.path.splitext(file_name)[0])
    name = name + '.png'
    print(name)
    print(recording)
    s, fs = sound.load(recording)
    s_trim = maad.sound.trim(s,fs, 0,3,pad=True,pad_constant=0)
    S = librosa.feature.melspectrogram(y=s_trim,  # recording as np array
                                      sr=fs,  # sampling rate
                                      n_mels=100,  # Number of mel bins
                                      fmax=fmax,  # Maximum frecuency
                                      fmin=fmin,
                                      n_fft=1024,  # length of the FFT window
                                      window=signal.windows.hann(1024),
                                      power=2)
    list_nparray.append([S])  
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB,sr=s,fmax=6000, ax=ax)
    list_png.append([fig])
    plt.savefig(fname=name)
    plt.close()
