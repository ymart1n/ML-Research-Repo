# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:49:16 2020

@author: Fan
"""
import numpy as np
import pylab as pl
import statsmodels.api as sm
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(x,theta):
    return sigmoid(np.dot(x, theta.T))

class LogReg:
    
    def init(self):
        np.random.seed(1)
        self.weight = np.random.random((3,1)) - 1
    
    def sigmoidDerivative(x):
        return x * (1 - x)    
    
    def cost(x, y, theta):
        likelihood = np.multiply(-y, np.log(model(x,theta))) - np.multiply(1 - y,np.log(1 - model(x, theta)))
        return np.sum(likelihood)/len(x)
        
    def gradient(x, y , theta):
        grad = np.zeros(theta.shape)
        error = (model(x,theta) - y).ravel()
        for i in range(len(theta.ravel())):
            term = np.multiply(error, x[:,i])
            grad[0, i] = np.sum(term) / len(x)
            
        return grad  