# Import default packages
import pandas as pd
import numpy as np
from datetime import datetime
#from numpy import nanmean, nanstd
import scipy.stats as stats


# Import default visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for string handling
from rapidfuzz import fuzz
import re

# Import files handling
#from google.colab import files
import io
import warnings
import joblib


# Import scalers and transformators
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

#Import Anova
from scipy.stats import f_oneway

# Import for splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Metrics for evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import explained_variance_score

# Import Statsmodels for Linear Regression
import statsmodels.api as sm

#Import lightGBM model
import lightgbm as lgbm

#import XGBOOST model
import xgboost as xgb

#import RandomForest model
from sklearn.ensemble import RandomForestRegressor

# Import PCA, MCA, FAMD
import prince
from prince import FAMD

# Import Tuning and Feature Selection
from sklearn.model_selection import GridSearchCV



warnings.filterwarnings("ignore")

#streamlit impoty
import streamlit as st

#handling url error
from urllib.error import URLError

#profiling
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport


##########################################################################################################
##################################SIDE BAR DEFINITION#####################################################
##########################################################################################################

'''
Section that customizes the lateral side bar with image and  info about the app
'''

st.sidebar.image('static/yin-yan-cats.jpg',width=120)

with st.sidebar.expander('About the app'):
   st.write("""
            This is the first DS project using Streamlite
                """)
   

##########################################################################################################
##################################MAIN AREA DEFINITION####################################################
##########################################################################################################

# Title customized based on HTML   
st.markdown(""" <style> .font {                                          
    font-size:30px ; font-family: 'Cooper Black'} 
    </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Data Profiling</p>', unsafe_allow_html=True)



# Function to deal with data on the given format

def converte_mes(data) -> str:
  '''
    Function that receives the data into a complete string format and converts into MM DD YYYY as string

    ----------------------
    Parameters
        str: string of date as
            'Tue Dec 16 2014 12:30:00 GMT-0800 (PST)'
    ----------------------
    Returns
        str: string on format MM DD YYYY as 
            '01 01 1900'

  '''
  mes = data[0:3]
  match mes:
    case 'Jan': mes = '01'
    case 'Feb': mes = '02'
    case 'Mar': mes = '03'
    case 'Apr': mes = '04'
    case 'May': mes = '05'
    case 'Jun': mes = '06'
    case 'Jul': mes = '07'
    case 'Aug': mes = '08'
    case 'Sep': mes = '09'
    case 'Oct': mes = '10'
    case 'Nov': mes = '11'
    case 'Dec': mes = '12'
    case default: mes = '00'

  if mes == '00':
    return '01 01 1900'
  else: return mes + data[3:11]



@st.cache_data

def get_car_data() -> pd.DataFrame:
    '''
    Function that reads the CSV file and transforms into a pandas DataFrame
    ----------------------
    Parameters
        None
    ----------------------
    Returns
        dataframe: csv file as pandas dataframe
    '''
    df = pd.read_csv('car_prices.csv')    
    return df
try:
    df = get_car_data()
except URLError as e:
   st.error(
    """
        Connection error: %s
    """
    % e.reason
   )


# variables choice on sidebar
option1=st.sidebar.radio(
     'What variables do you want to include in the Data Profiling?',
     ('All variables', 'A subset of variables'))

if option1=='All Variables':
     df=df

elif option1=='A subset of variables':
        var_list=list(df.columns)
        option2=st.sidebar.multiselect(
            'Select variable(s) you want to include in the report.',
            var_list)
        df=df[option2]

#dropdown menu with the minimal or complete mode
option3 = st.sidebar.selectbox(
     'Choose Minimal Mode or Complete Mode',
     ('Minimal Mode', 'Complete Mode'))

if option3=='Complete Mode':
    mode='complete'
    st.sidebar.warning('The default minimal mode disables expensive computations such as correlations and duplicate row detection. Switching to complete mode may cause the app to run overtime or fail for large datasets due to computational limit.')
elif option3=='Minimal Mode':
    mode='minimal'

# display of the dataframe on screen
st.dataframe(df)



#generate profiling based on choice of sidebar
if st.button('Generate Profiling'):
        if mode=='complete':
            profile=ProfileReport(df,
                title="User uploaded table",
                progress_bar=True,
                dataset={
                    "description": 'This profiling report was generated by Mau',
                    "copyright_holder": 'Mau',
                    "copyright_year": '2024'
                }) 
            st_profile_report(profile)
        elif mode=='minimal':
            profile=ProfileReport(df,
                minimal=True,
                title="User uploaded table",
                progress_bar=True,
                dataset={
                    "description": 'This profiling report was generated by Mau',
                    "copyright_holder": 'Mau',
                    "copyright_year": '2024'
                }) 
            st_profile_report(profile)  