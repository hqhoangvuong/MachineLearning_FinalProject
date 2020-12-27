# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:42:19 2020

@author: Hikari
"""
from data_puller import DataPuller
from data_processing import DataProcessing
from trainer import Trainer

def main():
    data_puller = DataPuller('https://cloudcomputing-finalproj-hkr.firebaseio.com/', '/DHT11')
    df = data_puller.getData()
    
    data_processor = DataProcessing(df)
    X_train, X_test, y_train, y_test = data_processor.getSet(time_interval_error_rate = 900, feature_scale = True)
    
    trainer = Trainer(X_train, y_train, X_test, y_test)
    liner_model = trainer.trainLinearRegression()
    ridge_model = trainer.trainRidgeRegression()
    lasso_model = trainer.trainLassoRegression()
    enet_model = trainer.trainEnetRegression()
    
if __name__ == '__main__':
    main()
    