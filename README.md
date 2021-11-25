# ML-Project--Used-cars-Price-Prediction
by **Anmol Jindal, Dhroov Goel, Sakshat Mali and Shubham Garg, Machine Learning (CSE343, ECE343)** from **Indraprastha Institute of Information Technology, Delhi**.

### Introduction
Predicting the price of second-hand cars is not an easy task.
In recent times, there has been an increase in the demand
for used cars, making it essential to predict the price more
accurately due to consumer exploitation, which can happen
due to high demand. <br>
Predicting the price of a car includes various factors like
mileage, version, location, engine capacity, power steering,
and many more. Collating this information and buying the
perfect fit can take some time and even might not produce
fruitful results.


### Description and Implementation Details
For this project, we have implemented various machine learning aglorithms on the dataset extracted from kaggle. We have used the following steps to reach to our conclusion - <br>

1. Data Cleaning and Preprocessing - In this we first did feature selection, then we encoded the categorical features. Then we handled the null values by either dropping the rows containing the null values or by replacing null values by method of imputation. 
3. Methodology - For this we used various machine learning techniques that best suited our model. These include Linear Regression, Ridge regression, Lasso regression, KNN, Random Forest, AdaBoost, XGBoost. 

### How to Run ? 


### Installed dependencies

  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import missingno as ms
  from sklearn.preprocessing import LabelEncoder
  from tqdm import tqdm
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  from sklearn.impute import SimpleImputer
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.linear_model import BayesianRidge
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn.preprocessing import OrdinalEncoder
  from sklearn.model_selection import cross_val_score
  from sklearn.model_selection import RepeatedStratifiedKFold
  from sklearn.ensemble import AdaBoostRegressor
  from sklearn.pipeline import Pipeline
  from sklearn.metrics import mean_squared_log_error,r2_score,mean_squared_error
  from sklearn.linear_model import Ridge
  from sklearn.linear_model import LassoCV,RidgeCV
  from yellowbrick.regressor import AlphaSelection
  import xgboost as xgb
  from sklearn.linear_model import Lasso
  from sklearn.neighbors import KNeighborsRegressor
  import warnings
  warnings.filterwarnings('ignore')

### Contact 
For any furhter queries feel free to reach out the following contributors. 

Anmol Jindal (anmol19294@iiitd.ac.in) </br>
Droov Goel (droov19303@iiitd.ac.in) </br>
Sakshat Mali (sakshat19327@iiitd.ac.in) </br>
Shubham Garg (shubham19336@iiitd.ac.in) </br>
