# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:42:33 2020

@author: Hikari
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessing:
    data_frame = ''
    threshold = 0
    df_future = pd.DataFrame(columns = ['Humidity', 'Temperature', 'UnixTime', 'Hour', 'Minute', 
                                          'Humidity_Future', 'Temperature_Future', 'UnixTime_Future', 'Hour_Future', 'Minute_Future'])
    
    def __init__(self, df):
        self.data_frame = df
        
    def getFuture(self, future_threshold):
        self.threshold = future_threshold
        for i, j in self.data_frame.iterrows():
            input = j['UnixTime'] + self.threshold
            feature = self.data_frame.iloc[(self.data_frame['UnixTime'] - input).abs().argsort()[:2]]
            selected = feature.iloc[0]
            
            if selected['UnixTime'] == j['UnixTime']:
                break
            
            if abs(selected['UnixTime'] - j['UnixTime'] - self.threshold) > abs(feature.iloc[1]['UnixTime'] - j['UnixTime'] - self.threshold):
                selected = feature.iloc[1]
                
            self.df_future = self.df_future.append({'Humidity' : j['Humidity'], 
                                                    'Temperature' : j['Temperature'], 
                                                    'UnixTime' : j['UnixTime'], 
                                                    'Hour' : j['Hour'], 
                                                    'Minute' : j['Minute'], 
                                                    'Humidity_Future' : selected['Humidity'], 
                                                    'Temperature_Future' : selected['Temperature'], 
                                                    'UnixTime_Future' : selected['UnixTime'], 
                                                    'Hour_Future' : selected['Hour'], 
                                                    'Minute_Future' : selected['Minute']},  
                                                   ignore_index = True)
        return self.df_future
    
    def verify_interval(self):
        verify_time = []
        for i, j in self.df_future.iterrows():
            time_dist = abs(j['UnixTime_Future'] - j['UnixTime'] - 60 * 60) 
            verify_time.append(time_dist)
        
        print(np.max(verify_time))
        n, bins, patches = plt.hist(verify_time)
        print('n: {0} \n bins: {1} \n patches: {2}'.format(n, bins, patches))
        plt.show()
        
    def pureData(self, threshold):
        for i, j in self.df_future.iterrows():
            if abs(j['UnixTime_Future'] - j['UnixTime'] - self.threshold) >= threshold:
                self.df_future.drop(index = i, inplace = True)
                
    def getSet(self, time_interval = 21600, time_interval_error_rate = 100, feature_scale = False, test_size = 0.2):
        self.getFuture(time_interval)
        self.pureData(time_interval_error_rate)
        X = self.df_future[['Humidity', 'Temperature', 'Hour', 'Minute']]
        y = self.df_future['Temperature_Future']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        
        if feature_scale:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[X_train.columns]), columns = X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test[X_test.columns]), columns = X_test.columns)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            return X_train, X_test, y_train, y_test
        
        
        