# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:09:37 2020

@author: Hikari
"""

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

class Trainer:
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    linear_regression_model = LinearRegression(n_jobs = -1) 
    ridge_regression_model = Ridge()
    lasso_regression_model = Lasso(alpha = 0.01)
    enet_model = ElasticNet(alpha = 0.01)
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def evaluateModel(self, model):
        pred_train = model.predict(self.X_train)
        print('Evaluate on train set:')
        print('Mean squared error: {0}'.format(np.sqrt(mean_squared_error(self.y_train, pred_train))))
        print('R2 Score: {0}'.format(r2_score(self.y_train, pred_train)))
        print('\n')
        
        pred_test= model.predict(self.X_test)
        print('Mean squared error: {0}'.format(np.sqrt(mean_squared_error(self.y_test, pred_test))))
        print('R2 Score: {0}'.format(r2_score(self.y_test, pred_test)))
        
    def trainLinearRegression(self):
        self.linear_regression_model.fit(self.X_train, self.y_train)
        self.evaluateModel(self.linear_regression_model)
        return self.linear_regression_model
    
    def trainRidgeRegression(self):
        self.ridge_regression_model.fit(self.X_train, self.y_train)
        self.evaluateModel(self.ridge_regression_model)
        return self.ridge_regression_model
    
    def trainLassoRegression(self, alpha_regu = 0.01):
        if alpha_regu != 0.01:
            self.lasso_regression_model = Lasso(alpha = alpha_regu)
        self.lasso_regression_model.fit(self.X_train, self.y_train)
        self.evaluateModel(self.lasso_regression_model)
        return self.lasso_regression_model
    
    def trainEnetRegression(self, alpha_regu = 0.01):
        if alpha_regu != 0.01:
            self.enet_model = ElasticNet(alpha = alpha_regu)
        self.enet_model.fit(self.X_train, self.y_train)
        self.evaluateModel(self.enet_model)
        return self.enet_model