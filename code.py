import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from statistics import mean
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score

#Kaggle Competition
#URL: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

working_path = "~/Documentos/github/kaggle_store_sales/datasets/"


holi_dataset = pd.read_csv(working_path + 'holidays_events.csv')
train_dataset = pd.read_csv(working_path + 'train.csv')
test_dataset = pd.read_csv(working_path + 'test.csv')
transaction_dataset = pd.read_csv(working_path + 'transactions.csv')
stores_dataset = pd.read_csv(working_path + 'stores.csv')
oil_dataset = pd.read_csv(working_path + 'oil.csv')
sample_sub_dataset = pd.read_csv(working_path + 'sample_submission.csv')

stores_dataset.sample(10)
oil_dataset.sample(10)
holi_dataset.sample(10)
stores_dataset.sample(10)

holi_dataset.describe()

train_dataset.sample(10)
test_dataset.sample(10)
sample_sub_dataset.sample(10)
transaction_dataset.sample(10)

#USED DATASETS:
    #holi_dataset
    #train_dataset
    #stores_dataset
    #oil_dataset
    #test_dataset

