import streamlit as st
# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Utils
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, \
                                          classification_report, confusion_matrix, plot_confusion_matrix
from scipy.stats import loguniform

# Preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from tensorflow.keras import layers, models

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

st.title("Hello world, and welcome to People Analytics!")
st.sidebar.title("People Analytics - here for you.")
st.markdown(" ## We specialize in employee analytics with AI.")
st.sidebar.markdown("This application will show you our analytics in order to ")

#importando DataFrame
people = pd.read_csv('raw_data/people_feat_eng.csv')
X = people.drop('attrition', axis = 1)
y = people['attrition']


