#-----------------EXTRACTING LABELLED CALLS AS SPECTROGRAMS-------------------------------------------------
#Required packages
import glob
import pandas as pd
from pathlib import Path
import numpy as np
import maad
from maad import sound
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import csv
import os
#Variables to define 
path_of_the_txt='C:/Users/sruizguz/txt_for_label/txts'
path_of_the_wavs='C:/Users/sruizguz/Labeling project recordings'

txt_list=[]
fin = pd.DataFrame()
list_nparray= []
list_png= []

# Select every .txt in the folder
for txt in glob.iglob(f'{path_of_the_txt}/*.txt'): 
    
    # Read the frequency and time values
    dfm = pd.read_csv(txt, sep='\t',
                     usecols=["Begin Time (s)","End Time (s)","Begin Time (s)","Low Freq (Hz)","High Freq (Hz)","Delta Time (s)","Delta Freq (Hz)"])
                      # usecols for selecting the columns of interest
    # Isolating the file name from the path-------------------------------------------  
    file_name= os.path.basename(txt)
    # Filename without extension (.txt in this case)
    name = (os.path.splitext(file_name)[0])
    
    # New column fill it with the file name
    dfm['f_name']= name
    #Removing the "_AP" (initials of the labeller)
    dfm['f_name'] = dfm['f_name'].map(lambda x: str(x)[:24]) # the "[:24]" selecth the character   
    #------------------------------------------------------------------------------------
    # Read and separate the Labels in a independent dataframe
    label = pd.read_csv(txt, sep=",")
    label['Species']= label.iloc[:, 0].str[-6:]
    
    #Join both dataframes (The one with the data measured in Raven (dfm) with the species data (label[Species]))
    dfm=pd.concat([dfm, label['Species']], axis=1)
    dfm['Species'] = dfm['Species'].astype(str)
    dfm = dfm[dfm.Species != str]
    
    #-------------------------------------------------------------------------------------
    #Adding the new extension (.wav)
    dfm['wav']= dfm['f_name'] + '.wav'
    # Change the structure of the new wav column 
    dfm['wav'] = dfm['wav'].astype(str)
    
    # Change the structure to numeric of the End time and Begin time column
    dfm['End Time (s)'] = pd.to_numeric(dfm['End Time (s)'])
    dfm['Begin Time (s)'] = pd.to_numeric(dfm['Begin Time (s)'])
    
    #Filename for the spectrograms-------------------------------------------------------
    # Create a column with the filename plus the .png extension
    dfm['png']= dfm['f_name'] + '.png'  
    # Create a column with a sequence from 1 to number of rows
    dfm['row_num'] = np.arange(len(dfm))   
    # Create a column merging the filename (png) plus the sequence
    dfm["title"] = dfm['Species'].astype(str) + '-'+ dfm['row_num'].astype(str) + '-'+ dfm['png'].astype(str)
    
    # Creating a column with the names of the .WAV paths
    dfm['filepath']=path_of_the_wavs+'/'+ dfm['wav']
    #Drop out unecessary columns
    dfm.drop(['row_num','png'], axis=1, inplace=True)
    
    # Loop for iterating over every row inside the text file---------------------------------------------------------
    for index, row in dfm.iterrows():
        if not os.path.isfile(row['filepath']):    # If the filepath is actually a file, otherwise skip that iteration
            continue                               
        txt_list.append(row['filepath'])               
    # Selecting each row in every .txt file----------------------------------------------------------------------------
    #for index, row in dfm.iterrows():
        # Read the .wav as an array (s) and getting the sample rate (fs)
        s, fs = librosa.load(row['filepath']) 
        # Selecting an specific section of the recording based on the rows
        s_trim = maad.sound.trim(s,fs, min_t=row['Begin Time (s)'], #The minimun time is the value in the Begin time value
                                 max_t=row['End Time (s)'],         #The maximun time is the value in the End time value
                                 pad=True,pad_constant=0)
        # Mel-scale spectrogram for the vocalizations
        mel = librosa.feature.melspectrogram(y=s_trim,  # recording as np array
                                      sr=fs,            # sampling rate
                                      n_mels=100,       # Number of mel bins
                                      fmax=6000,        # Maximum frequency
                                      fmin=2000,        # Minimun frequency
                                      n_fft=1024,       # length of the FFT window
                                      window=signal.windows.hann(1024),
                                      power=2)    
        list_nparray.append([mel])  #Append every iteration to a list of numpy arrays
        fig, ax = plt.subplots()    
        S_dB = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(S_dB,sr=s,fmax=6000, ax=ax)
        list_png.append([fig])
        plt.savefig(fname=row['title'])
        plt.close()     
    
