# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:42:33 2020

@author: Hikari
"""

import pandas as pd
import numpy as np
from firebase import firebase
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os

class DataPuller:
    firebase_application_url = ''
    firebase_realtimedb_path = ''
    firebase_app = ''
    data_frame = pd.DataFrame()
    
    def __init__(self, application_url, db_path):
        self.firebase_application_url = application_url
        self.firebase_realtimedb_path = db_path
        self.firebase_app = firebase.FirebaseApplication(self.firebase_application_url)
        result = self.firebase_app.get(self.firebase_realtimedb_path, None)
        self.data_frame = pd.DataFrame.from_dict(result).T
        
    def pureData(self):
        self.data_frame['Humidity'] = self.data_frame['Humidity'].str.strip().str.replace('"','').astype(float)
        self.data_frame['Temperature'] = self.data_frame['Temperature'].str.strip().str.replace('"','').astype(float)
        self.data_frame['UnixTime'] = self.data_frame['UnixTime'].str.strip().str.replace('"','').astype(np.int64)
        self.data_frame.dropna(how = 'all', inplace = True)
        self.data_frame['Humidity'].fillna(self.data_frame['Humidity'].mean(), inplace = True)
        self.data_frame['Temperature'].fillna(self.data_frame['Temperature'].mean(), inplace = True)
        self.data_frame.reset_index(drop = True, inplace = True)

    def proccessData(self):
        self.pureData()
        for i, j in self.data_frame.iterrows():
            if j['Temperature'] > 35.0:
                self.data_frame.drop(i, inplace = True)
        self.data_frame.reset_index(drop = True, inplace = True)
        
        hour_list = []
        minute_list = []

        for i, j in self.data_frame.iterrows():
            hour = int(time.strftime("%H", time.gmtime(int(j['UnixTime']))))
            minute = int(time.strftime("%M", time.gmtime(int(j['UnixTime']))))
            hour_list.append(hour)
            minute_list.append(minute)
        
        self.data_frame.insert(3, 'Hour', hour_list)
        self.data_frame.insert(4, 'Minute', minute_list)
        return self.data_frame
        
    def mkdir(self):
        dir_saved_dataframe = os.path.join('saved_dataframe')
        dir_figs = os.path.join('figs')
        
        if not os.path.exists(dir_saved_dataframe):
            os.mkdir(dir_saved_dataframe)
            
        if not os.path.exists(dir_figs):
            os.mkdir(dir_figs)
        
    def saveData(self, filename):
        self.mkdir()
        path = r'saved_dataframe/df_pure_19_12_20.csv'
        
        if(filename):
            path = r'saved_dataframe/{0}'.format(filename)
            
        self.data_frame.to_csv(path, index=False, header=True)
        print('Saved dataframe at {0}'.format(path))
        
    def heatmap(self, data, figname = 'untitle', figtitle = 'untitle'):
        self.mkdir()
        plt.figure(figsize=(35, 20))
        sns.heatmap(data, cmap='coolwarm', annot=True)
        filename = 'figs/{0}.png'
        plt.title(figname)
        plt.savefig(filename.format(figname))
        plt.show()
        
    def heatmap_temp(self):
        temp_hour = pd.crosstab(self.data_frame['Hour'], self.data_frame['Minute'], values = self.data_frame['Temperature'], aggfunc = 'mean')
        self.heatmap(temp_hour, 'avg_temp_hourly', 'Average Tempearture in HCMC from 0h to 23h')
    
    def heatmap_hum(self):
        hum_hour = pd.crosstab(self.data_frame['Hour'], self.data_frame['Minute'], values = self.data_frame['Humidity'], aggfunc='mean')
        self.heatmap(hum_hour, 'avg_hum_hourly', 'Average Humidity in HCMC from 0h to 23h')
    
        
    
        
    