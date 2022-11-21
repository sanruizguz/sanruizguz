import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
import sys
import os
import time
import warnings
# suppress all warnings
warnings.filterwarnings("ignore")

from scipy import signal
import librosa
import librosa.display
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from maad import sound, util, rois

def grab_audio(path, audio_format='mp3'):
    filelist = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name[-3:].casefold() == audio_format and name[:2] != '._':
                filelist.append(os.path.join(root, name))
    return filelist
  
  # Folder where I will keep the Xenocanto files after downloading
XC_ROOTDIR = 'J:/Xenocanto/'

XC_DIR = 'USF&W_3_dataset'

data = [['Bachmans Sparrow', 'Peucaea aestivalis'],
        ['Carolina Wren','Thryothorus ludovicianus'],
        ['Eastern Towhee','Pipilo erythrophthalmus'],]

df_species = pd.DataFrame(data,columns =['english name',
                                         'scientific name'])
gen = []
sp = []
for name in df_species['scientific name']:
    gen.append(name.rpartition(' ')[0])
    sp.append(name.rpartition(' ')[2])
    
    
 df_query = pd.DataFrame()
df_query['param1'] = gen
df_query['param2'] = sp
#df_query['param3'] ='type:song'
#df_query['param4'] ='area:america'
# df_query['param5 ='len_lt:120'
# df_query['param6'] ='len_gt:5'
# df_query['param7'] ='q_gt:C'

# Get recordings metadata corresponding to the query
df_dataset= util.xc_multi_query(df_query,
                                 format_time = False,
                                 format_date = False,
                                 verbose=True)

df_dataset = util.xc_selection(df_dataset,
                               max_nb_files=600,
                               max_length='10:00',
                               min_length='00:03',
                               min_quality='E',
                               verbose = True )
util.xc_download(df_dataset,
                 rootdir = XC_ROOTDIR,
                 dataset_name= XC_DIR,
                 overwrite=True,
                 save_csv= True,
                 verbose = True)

filelist=[]
path = "J:/Xenocanto/USF&W_3_dataset"

for root, directories, files in os.walk(path, topdown=False):
    for name in files:
        lista=os.path.join(root, name)  
        filelist.append(lista)
        
print(filelist)

filelist=[]
path = "J:/Xenocanto/USF&W_3_dataset/BACS"

for root, directories, files in os.walk(path, topdown=False):
    for name in files:
        lista=os.path.join(root, name)  
        filelist.append(lista)
        
print(filelist)

df = pd.DataFrame()
for file in filelist:
    df = df.append({'fullfilename': file,
                    'filename': Path(file).parts[-1][:-4], 
                    'species': Path(file).parts[-2]},
                     ignore_index=True)

print('=====================================================')
print('number of files : %2.0f' % len(df))
print('number of species : %2.0f' % len(df.species.unique()))
print('=====================================================')
