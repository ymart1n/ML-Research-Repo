# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 02:22:05 2020

@author: Fan
"""
import time
import numpy as np
import pylab as pl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
  datapath = 'binary.csv'    
  training_data = pd.read_csv(datapath,skiprows=(0),nrows=(300))
  predict_data = pd.read_csv(datapath,skiprows=(300),nrows=(100))
  print (training_data.head())
  
  training_data.columns = ["admit","gre","gpa","ranking"]
  predict_data.columns = ["admit","gre","gpa","ranking"]
  
  train_cols = training_data.loc[0:,["gre","gpa","ranking"]]
  pred_cols = predict_data.columns[1:]
  
  start = time.time()

  clf = RandomForestClassifier(n_estimators=10)
  clf.fit(train_cols, training_data.loc[0:,["admit"]])
  predict_data['predict'] = clf.predict(predict_data[pred_cols])
  
  end = time.time()
  print("Time:%f" %(end - start))
  
  total = 0
  hit = 0
  for value in predict_data.values:
      total += 1
      predict = value[-1]
      admit = int(value[0])
 
  # 假定预测概率大于0.5则表示预测被录取
      if(predict >= 0.5 and admit ==1):
         hit += 1
      elif(predict < 0.5 and admit ==0):
         hit += 1
         
  print (predict_data.head())
  print("Total: %d, Hit: %d, Precision: %.2f" % (total, hit, 100.0*hit/total) )
  
if __name__ == "__main__":  
	main()
	print("All finished!")

