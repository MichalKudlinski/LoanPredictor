"""Choosing the right dataset for the job:
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

"""

"""Importing the necessary libraries and modules"""

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class DataProvider():
   def read_csv(self,csv_file, csv_file1):
      global data_frame,data_frame1
      data_frame = pd.read_csv(csv_file)
      data_frame1 = pd.read_csv(csv_file1)
      

dp = DataProvider()
dp.read_csv('train_u6lujuX_CVtuZ9i.csv','test_Y3wMUE5_7gLdaTN.csv')

"""Visualizating data from training set"""
class DataVisualizator():
   """Class for visualizing dataframe"""
   def show_head(self,df):
      print( df.head())
   def show_shape(self,df):
      print( df.shape)
   def describe(self,df):
      print(df.describe())
   def is_null(self,df):
      print(df.isnull().sum())
   def info(self,df):
      print(df.info())
  
      



dv= DataVisualizator()
# dv.show_head(data_frame)
# dv.show_shape(data_frame)
# dv.show_labels(data_frame)
# dv.is_null(data_frame)
# dv.info(data_frame)

"""Manipulating data in trainingn and test datasets"""

class DataManipulator():
     """Class for manipulating data in dataframe """
     def drop_column(self,df,column_name,axis,inplace):
      df.drop(column_name,axis,inplace=True)

     def drop_null_rows(self,df,inplace):
      df.dropna(inplace = True)

     def to_numeric_converter(self,df,df1):
         to_numeric = {'Male': 1, 'Female': 2,
         'Yes': 1, 'No': 2,
         'Graduate': 1, 'Not Graduate': 2,
         'Urban': 3, 'Semiurban': 2,'Rural': 1,
         'Y': 1, 'N': 0,
         '3+': 3}

         global train_dataframe, test_dataframe
         train_dataframe = df.applymap(lambda x: to_numeric.get(x) if x in to_numeric else x)
         test_dataframe = df1.applymap(lambda x: to_numeric.get(x) if x in to_numeric else x) 

   
   
dm = DataManipulator()
#deleting loan_id column as it is useless for our modeling
dm.drop_column(data_frame,'Loan_ID',axis = 1, inplace = True)
dm.drop_column(data_frame1,'Loan_ID',axis = 1, inplace = True)
#deleting rows with any null values from both datasets
dm.drop_null_rows(data_frame, inplace = True)
dm.drop_null_rows(data_frame1, inplace = True)
dv.is_null(data_frame)
dv.show_head(data_frame)
dv.info(data_frame)  
#transforming all data types to numeric 
dm.to_numeric_converter(data_frame,data_frame1)

#creating correlation matrix 
corr_matrix = train_dataframe.corr()
sn.heatmap(corr_matrix,annot = True)
plt.show()

#using logistic regression and linear regression models 


class ModelTrainer():

   def training_prep(self,df):
      global X_train, X_test, Y_train, Y_test, X, Y
      Y = df['Loan_Status']
      X = df.drop('Loan_Status', axis = 1)
      global X_train, X_test, Y_train, Y_test 
      X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

   
   def logistic_reg(self):

      global LR_SC
      
      LR = LogisticRegression()
      LR.fit(X_train, Y_train)

      Y_predict = LR.predict(X_test)

      LR_SC = LR.score(X_test, Y_test)

   def decision_tree(self):
      global DT_SC

      DT = DecisionTreeClassifier()
      DT.fit(X_train, Y_train)

      Y_predict = DT.predict(X_test)

      DT_SC = accuracy_score(Y_predict,Y_test)
   def random_forest(self):
      global RF_SC
      RF = RandomForestClassifier()
      RF.fit(X_train, Y_train)

      Y_predict = RF.predict(X_test)


   
      RF_SC = accuracy_score(Y_predict,Y_test)


      

Mt = ModelTrainer()
Mt.training_prep(train_dataframe)
Mt.logistic_reg()
Mt.decision_tree()
Mt.random_forest()
print(LR_SC)
print(DT_SC)
print(RF_SC)

class ScoreChecker():
   def train_100000_times(self):
      global result
      sum_LR = 0
      sum_RF = 0
      for i in range(100):
         Mt.logistic_reg()
         sum_LR+= LR_SC
      for i in range(100):
         Mt.random_forest()
         sum_RF+= RF_SC
      result = (sum_LR, sum_RF)

SC = ScoreChecker()
SC.train_100000_times()
print(result)


   






                                              

