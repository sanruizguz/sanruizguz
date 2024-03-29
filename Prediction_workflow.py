import matplotlib as plt
import numpy as np
import os
import glob
import librosa.display
import maad
from maad import sound
from scipy import signal



def openAudioFile(path, sample_rate=48000, offset=0.0, duration=None):    
    import librosa
    try:
        sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type='kaiser_fast')
    except:
        sig, rate = [], sample_rate
    return sig, rate

def splitSignal(sig, rate, seconds, overlap, minlen):
    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]
        # End of signal?
        if len(split) < int(minlen * rate):
            break       
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, noise(split, (int(rate * seconds) - len(split)), 0.5)))        
        sig_splits.append(split)
    return sig_splits

def getRawAudioFromFile(fpath):
    # Open file
    sig, rate = openAudioFile(fpath, SAMPLE_RATE)
    # Split into raw audio chunks
    chunks = splitSignal(sig, rate, SIG_LENGTH, SIG_OVERLAP, SIG_MINLEN)
    return chunks
 
    
def noise(sig, shape, amount=None):
    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0, 0)
    # Create Gaussian noise
    try:
        noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)
    except:
        noise = np.zeros(shape)
    return noise.astype('float32')
  
  
  # Sample rate
SAMPLE_RATE=48000

# Length of the sliced recordings in seconds
SIG_LENGTH = 3

# Overlap
SIG_OVERLAP= 0

# Lentgh of the original recordings in minutes, for seconds use decimals ej. 0.30=30 seconds
SIG_MINLEN = 0.59

# Number of spectrograms generated by recording
SPECS_PER_RECORD= 20

RANDOM_SEED= 42
RANDOM = np.random.RandomState(RANDOM_SEED)

path_of_the_directory='C:/Users/Default/Desktop/prueba'

fmin = 2000
fmax = 6000


import pandas as pd
import matplotlib as plt
from numpy import asarray
from PIL import Image
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path

new_col= ['Prediction']
new_empt_list= []

new_name= ['File']
new_name_file= []

new_spec= ['Spectro']
new_spectro= []


#ARGMAX
argmaxlist=[]


for recording in glob.iglob(f'{path_of_the_directory}/*.wav'): # Select each recording in the folder
    getaudio= getRawAudioFromFile(recording)  # Load and split the recording in 3 second recordings 
    for x in getaudio:
        S = librosa.feature.melspectrogram(y=x,  # recording as np array
                                   sr=48000,  # sampling rate
                                   n_mels=100,  # Number of mel bins
                                   fmax=2000,  # Maximum frecuency
                                   fmin=6000,  # Minimun frecuecy
                                   n_fft=1024,  # length of the FFT window
                                   window=signal.windows.hann(1024),
                                   power=2)
        res = cv2.resize(S, dsize=(255, 255), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(res, 'RGB')
        batch = np.expand_dims(img, axis=0)
        img_preprocessed = preprocess_input(batch)
        prediction=model.predict(img_preprocessed)
        new_name_file.append([os.path.basename(recording)])
        new_empt_list.append([prediction])  
        
        argmax=np.argmax(prediction, axis=1)
        argmaxlist.append([argmax])
        
df = pd.DataFrame(new_empt_list, columns=new_col)
df2 = pd.DataFrame(new_name_file, columns=new_name)


def createList(n):
    lst = []
    for i in range(n):
        lst.append(i+1)
    return(lst)

wav_files = [f for f in os.listdir(path_of_the_directory) if f.endswith('.wav')]
len(wav_files)        
spectro = [createList(SPECS_PER_RECORD)]*len(wav_files)

flat_list = [item for sublist in spectro for item in sublist]

df2['Spectrogram']=np.array(flat_list)
df2['Prediction']=df
df2['Prediction'] = df2['Prediction'].map(lambda x: str(x)[2:])
df2['Prediction'] = df2['Prediction'].map(lambda x: str(x)[:-2])
df2[['Bachman','Carolina','Towhee']] = df2['Prediction'].str.split(n=2,expand=True)
df2['Bachman'] = pd.to_numeric(df2['Bachman'])
df2['Carolina'] = pd.to_numeric(df2['Carolina'])
df2['Towhee'] = pd.to_numeric(df2['Towhee'])
df2[1:20]

final_df=df2[['Bachman','Carolina','Towhee']].groupby(np.arange(len(df2))//20).mean()
final_df['File']= wav_files 
final_df = final_df[['File','Bachman','Carolina','Towhee']]
final_df

final_df['ARU']=final_df['File'].str[:8]
final_df['Date']=final_df['File'].str[9:17]
final_df['Time']=final_df['File'].str[18:24]
final_df = final_df[['File','ARU','Date','Time','Bachman','Carolina','Towhee']]
final_df.head()

       
