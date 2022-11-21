

# Edited from the example code from the Python package sckit-maad
# https://scikit-maad.github.io/_auto_examples/2_advanced/plot_extract_alpha_indices.html#sphx-glr-auto-examples-2-advanced-plot-extract-alpha-indices-py

import pandas as pd
import os
import numpy as np
import maad
from maad import sound, features
from maad.util import (date_parser, plot_correlation_map,
                       plot_features_map, plot_features, false_Color_Spectro)
import warnings
warnings.filterwarnings("ignore")

SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
'AGI','ROItotal','ROIcover']

TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
               'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
               'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount']

# Path where the recordings are located
df = date_parser(datadir="C:/Users/Default/Desktop/prueba",
                 dateformat='SM4', verbose=True)

#
df_indices = pd.DataFrame()
df_indices_per_bin = pd.DataFrame()


for index, row in df.iterrows() :

    # get the full filename of the corresponding row
    fullfilename = row['file']
    # Save file basename
    path, filename = os.path.split(fullfilename)
    print ('\n**************************************************************')
    print (filename)

    #### Load the original sound (16bits) and get the sampling frequency fs
    try :
        wave,fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)

    except:
        # Delete the row if the file does not exist or raise a value error (i.e. no EOF)
        df.drop(index, inplace=True)
        continue

    """ =======================================================================
                     Computation in the time domain
    ========================================================================"""

    # Parameters of the audio recorder. This is not a mandatory but it allows
    # to compute the sound pressure level of the audio file (dB SPL) as a
    # sonometer would do.
    S = -35         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)
    G = 26+16       # Amplification gain (26dB (SM4 preamplifier))

    # compute all the audio indices and store them into a DataFrame
    # dB_threshold and rejectDuration are used to select audio events.
    df_audio_ind = features.all_temporal_alpha_indices(wave, fs,
                                          gain = G, sensibility = S,
                                          dB_threshold = 3, rejectDuration = 0.01,
                                          verbose = False, display = False)

    """ =======================================================================
                     Computation in the frequency domain
    ========================================================================"""

    # Compute the Power Spectrogram Density (PSD) : Sxx_power
    Sxx_power,tn,fn,ext = sound.spectrogram (wave, fs, window='hann',
                                             nperseg = 1024, noverlap=1024//2,
                                             verbose = False, display = False,
                                             savefig = None)

    # compute all the spectral indices and store them into a DataFrame
    # flim_low, flim_mid, flim_hi corresponds to the frequency limits in Hz
    # that are required to compute somes indices (i.e. NDSI)
    # if R_compatible is set to 'soundecology', then the output are similar to
    # soundecology R package.
    # mask_param1 and mask_param2 are two parameters to find the regions of
    # interest (ROIs). These parameters need to be adapted to the dataset in
    # order to select ROIs
    df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(Sxx_power,
                                                            tn,fn,
                                                            flim_low = [0,1500],
                                                            flim_mid = [1500,8000],
                                                            flim_hi  = [8000,20000],
                                                            gain = G, sensitivity = S,
                                                            verbose = False,
                                                            R_compatible = 'soundecology',
                                                            mask_param1 = 6,
                                                            mask_param2=0.5,
                                                            display = False)

    """ =======================================================================
                     Create a dataframe
    ========================================================================"""
    # First, we create a dataframe from row that contains the date and the
    # full filename. This is done by creating a DataFrame from row (ie. TimeSeries)
    # then transposing the DataFrame.
    df_row = pd.DataFrame(row)
    df_row =df_row.T
    df_row.index.name = 'Date'
    df_row = df_row.reset_index()

    # add scalar indices into the df_indices dataframe
    df_indices = df_indices.append(pd.concat([df_row,
                                              df_audio_ind,
                                              df_spec_ind], axis=1))
    # add vector indices into the df_indices_per_bin dataframe
    df_indices_per_bin = df_indices_per_bin.append(pd.concat([df_row,
                                                              df_spec_ind_per_bin], axis=1))

    # Set back Date as index
df_indices = df_indices.set_index('Date')
df_indices_per_bin = df_indices_per_bin.set_index('Date')



#---------------------------------------24 HOURS CYCLE LINES------------------------------------------------------------
# A more classical way to analyse variations of indices consists in plotting
# graphs. We choose to normalize rescale their value between 0 to 1 in order to
# compare their trend during a 24h cycle
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['Hf']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['AEI']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['NDSI']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['ACI']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['MED']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['ROItotal']],norm=True,mode='24h', ax=ax[2,1])

#2
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['ZCR']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['MEANt']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['VARt']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['SKEWt']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['KURTt']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['LEQt']],norm=True,mode='24h', ax=ax[2,1])

#3
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['BGNt']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['SNRt']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['MED']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['ACTtCount']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['ACTtFraction']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['ACTtMean']],norm=True,mode='24h', ax=ax[2,1])

#4
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['EVNtCount']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['EVNtFraction']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['EVNtMean']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['MEANf']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['VARf']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['SKEWf']],norm=True,mode='24h', ax=ax[2,1])

#5
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['KURTf']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['NBPEAKS']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['LEQf']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['ENRf']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['BGNf']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['SNRf']],norm=True,mode='24h', ax=ax[2,1])


#6
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['EAS']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['ECU']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['ECV']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['EPS']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['EPS_KURT']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['EPS_SKEW']],norm=True,mode='24h', ax=ax[2,1])

#7
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['rBA']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['AnthroEnergy']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['BioEnergy']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['BI']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['ADI']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['LFC']],norm=True,mode='24h', ax=ax[2,1])
#8
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['MFC']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['HFC']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['ACTspFract']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['ACTspCount']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['ACTspMean']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['EVNspFract']],norm=True,mode='24h', ax=ax[2,1])
#9
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['EVNspMean']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['EVNspCount']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['TFSD']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['H_Havrda']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['H_Renyi']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['H_pairedShannon']],norm=True,mode='24h', ax=ax[2,1])
#10
fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))

fig, ax[0,0] = plot_features(df_indices[['H_gamma']],norm=True,mode='24h', ax=ax[0,0])
fig, ax[0,1] = plot_features(df_indices[['H_GiniSimpson']],norm=True,mode='24h', ax=ax[0,1])
fig, ax[1,0] = plot_features(df_indices[['RAOQ']],norm=True,mode='24h', ax=ax[1,0])
fig, ax[1,1] = plot_features(df_indices[['AGI']],norm=True,mode='24h', ax=ax[1,1])
fig, ax[2,0] = plot_features(df_indices[['ROItotal']],norm=True,mode='24h', ax=ax[2,0])
fig, ax[2,1] = plot_features(df_indices[['ROIcover']],norm=True,mode='24h', ax=ax[2,1])
#-----------------------------------FALSE COLOR SPECTROGRAMS------------------------------------------------------------
fcs, triplet = false_Color_Spectro(df_indices_per_bin,
                                   indices = ['KURTt_per_bin',
                                              'EVNspCount_per_bin',
                                              'MEANt_per_bin'],
                                   reverseLUT=False,
                                   unit='hours',
                                   permut=False,
                                   display=True,
                                   figsize=(4,7))

df_indices.to_csv(r'K:/USFW-SANTIAGO/plots/Prairie/Prairie_i.csv')
df_indices_per_bin.to_csv(r'K:/USFW-SANTIAGO/plots/Prairie/Prairie_ipb.csv')
