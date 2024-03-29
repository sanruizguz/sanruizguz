#------------------------------------------------------Required packages-----------------------------------------------------------------------
import librosa
import librosa.display
import soundfile
import glob
import os
from audiomentations import Compose, TimeMask, PitchShift, BandStopFilter, GainTransition,AddBackgroundNoise, FrequencyMask, AddGaussianNoise
import maad
from maad import sound
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------Augmentation settings-----------------------------------------------------------------
augment = Compose([
   AddBackgroundNoise(sounds_path="J:/Noise",
                      min_snr_in_db=-50,
                      max_snr_in_db=-40,
                      noise_rms="relative",
                      p=0.2),
    AddGaussianNoise(min_amplitude=0.01,
                     max_amplitude=0.02,
                     p=0.3),
   BandStopFilter( min_center_freq=500.0,
                   max_center_freq=10000.0,
                   min_bandwidth_fraction=0.1,
                   max_bandwidth_fraction=0.2,
                   p=0.2),
   TimeMask(min_band_part=0.1,
            max_band_part=0.2,
            fade=True,
            p=0.2),
   PitchShift(min_semitones=-1, max_semitones=1, p=0.1),
   GainTransition(min_gain_in_db=-50.0,max_gain_in_db=-40.0)
])

path_of_the_directory = "J:/AUGMENTATION_3SET/Original_plus_Xenocanto/E"
fmin = 2000
fmax = 6000

#------------------------------------------------------Augmented mel-scale spectrogram-----------------------------------------------------------------

list_png=[]
list_array=[]

for name in os.listdir(path_of_the_directory):
    f = os.path.join(path_of_the_directory, name)
    sr = librosa.get_samplerate(f)
    filename = os.path.basename(f)
    audio = librosa.load(f, duration=6,sr=None)
    augmented = augment(audio[0], sample_rate=sr)
    s_trim = maad.sound.trim(augmented,48000, 0,3,pad=True,pad_constant=0)
    S = librosa.feature.melspectrogram(y=s_trim,  # recording as np array
                                      sr=48000,  # sampling rate
                                      n_mels=100,  # Number of mel bins
                                      fmax=fmax,  # Maximum frecuency
                                      fmin=fmin,
                                      n_fft=1024,  # length of the FFT window
                                      window=signal.windows.hann(1024),
                                      power=2)
    list_array.append([S])  
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB,sr=sr,fmax=8000, ax=ax)
    list_png.append([fig])
    file_name = os.path.basename(name)
    name = (os.path.splitext(file_name)[0])
    name = name + '.png'
    print(name)
    plt.savefig(fname=name)
    plt.close()
