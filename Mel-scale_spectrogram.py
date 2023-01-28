#----------------Required packages for the Mel-scale spectrograms--------------
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import maad
from maad import sound
from scipy import signal


#---------------------Packages for noise filters----------------------------------
from maad.util import plot2d, power2dB
from maad.sound import (load, spectrogram,
                       remove_background, median_equalizer,
                       remove_background_morpho,
                       remove_background_along_axis, sharpness)
from timeit import default_timer as timer


# Minimun frequency threshold for the spectrograms
fmin = 2000
# Maximun frequency threshold for the spectrograms
fmax = 6000
# Path where the WAV files (labelled recordings are located)
path_of_the_directory="C:/Users/sruizguz/Downloads/Labeling project recordings-selected"

#Create 2 empty list before running the loop
list_nparray= []
list_png= []


#First, we need to locate each WAV file and create the spectrogram filename with the png extension
# For all the files (recordings) in the path which end in .wav:
for recording in glob.iglob(f'{path_of_the_directory}/*.wav'): 
    # Create a list with the full path of each .WAV in the directory
    file_name = os.path.basename(recording)
    # Split the WAV file name from the rest of the path 
    name = (os.path.splitext(file_name)[0])
    # Add the "png" extension to the filenames
    name = name + '.png'
    print(name)
    print(recording)
    
    # Then, will load each WAV as a numpy array using the sound.load function extracting the array (s) and the sampling rate (fs)
    s, fs = sound.load(recording)
    # Here I specified the desired recording lenght (initial=0, final=3seconds), in this way all the spectrograms will be 3s
    s_trim = maad.sound.trim(s,fs, 0,3,pad=True,pad_constant=0)
    # Create the melspectrograms:
    S = librosa.feature.melspectrogram(y=s_trim,  # recording as np array
                                      sr=fs,  # sampling rate
                                      n_mels=100,  # Number of mel bins       # Try different values to figure out the best spec visualization
                                      fmax=fmax,  # Maximum frecuency
                                      fmin=fmin,  # Minimun frequency
                                      n_fft=1024,  # length of the FFT window # Try different values to figure out the best spec visualization
                                      window=signal.windows.hann(1024),
                                      power=2)
    # Now, we will add each new spectrogram to the empty list we created before
    list_nparray.append([S])  
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Spectrogram settings (you can change some settings such as the palette of colors, axis etc)
    img = librosa.display.specshow(S_dB,sr=s,fmax=6000, ax=ax, cmap='gray_r')
    list_png.append([fig])
    # Then, we will name the new spec with the filename with the png extension
    plt.savefig(fname=name)
    # The new spec will be save by default in the C disk, check your User file.
    plt.close()
