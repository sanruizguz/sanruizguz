import glob
import matplotlib.pyplot as plt
from maad import sound, util

fpath = 'J:/Original_Labelled_Recordings/Bachmans sparrow/'  # location of audio files
sample_len = 1  # length in seconds of each audio slice

fs=44100

flist = glob.glob(fpath+'*.wav')
flist.sort()
long_wav = list()
for idx, fname in enumerate(flist):
    s, fs = sound.load(fname)
    s = sound.trim(s, fs, 0, sample_len)
    long_wav.append(s)
    
long_wav

long_wav = util.crossfade_list(long_wav, fs, fade_len=0.1)
Sxx, tn, fn, ext = sound.spectrogram(long_wav, fs,
                                     window='hann', nperseg=1024, noverlap=512)

fig, ax = plt.subplots(1,1, figsize=(10,3))
util.plot_spectrogram(Sxx, extent=[0, 24, 0, 11],
                      ax=ax, db_range=80, gain=25, colorbar=False)
ax.set_xlabel('Time [Hours]')
ax.set_xticks(range(0,25,4))
