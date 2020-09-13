import librosa
import csv
import pandas as p
import numpy as np
import librosa.display
import os
import matplotlib.pyplot as plt
base_file_location='E:\\birdsong\\train_audio\\'
base_save_location='E:\\birdsong\\train_mel\\'
panda_data=p.read_csv('E:\\birdsong\\train.csv')
counter=0
for index,row in panda_data.iterrows():
    counter+=1
    if(counter%100==0):
        print(counter)
    try:
        location=base_file_location+row['ebird_code']+'\\'+row['filename']
        sr=row['sampling_rate']
        y,sr=librosa.load(location,sr)
        S=librosa.feature.melspectrogram(y,sr=sr)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max), fmax=8000)
        if(not(os.path.isdir(base_save_location+row['ebird_code']+'\\'))):
            os.makedirs(base_save_location+row['ebird_code']+'\\')
            
        save_name=row['filename']
        save_name=save_name[:-4]
        save_name+='.jpg'
        plt.savefig(base_save_location+row['ebird_code']+'\\'+save_name)
        plt.close()
    except:
        print('fail at')
        print(counter)