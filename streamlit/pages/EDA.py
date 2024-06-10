# Import default packages
import pandas as pd
import numpy as np
from datetime import datetime
#from numpy import nanmean, nanstd
import scipy.stats as stats


# Import default visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

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

#wordcloud

from wordcloud import WordCloud



################################## SIDE BAR DEFINITION #####################################################


#Section that customizes the lateral side bar with image and  info about the app

st.sidebar.image('streamlit/static/ying-yang-cats.jpg',width=120)

with st.sidebar.expander('About the app'):
   st.write("""
            This is the first DS project using Streamlite
                """)
   


################################## MAIN AREA DEFINITION ####################################################

# Title customized based on HTML   
st.markdown(""" <style> .font {                                          
    font-size:30px ; font-family: 'Cooper Black'} 
    </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">EDA</p>', unsafe_allow_html=True)




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


df_nulls = pd.DataFrame(df.isnull().sum()).reset_index().rename(columns={'index':'column_name',0:'null_values'})
df_nulls['null_percents'] = df_nulls.null_values / len(df)
df_nulls = df_nulls.sort_values(by='null_percents', ascending = False)


######################################## Completeness Section ##############################################################


with st.expander('Completeness'):
    st.header('Completeness')
    st.subheader('How many null values each column has? How should we handle null values? Should we select features based on that?')


    #null_chart = st.bar_chart(df_nulls, x='column_name', y='null_percents')

    null_bar_plot = alt.Chart(df_nulls, title = "Column's Null Percentage").mark_bar().encode(
            alt.X('column_name').title('Column')
            ,alt.Y('null_percents').axis(format='.2%').title('Null Percentage')#.sort('-y')
            ,text = 'null_percents'
    ).properties(
    #width = 800,
    #height = 400
     width = 600
     ,height = 400
    )

    st.altair_chart(null_bar_plot)


    st.text('Based on relatively low null percentages, no column will be dropped. However, since the dataset is already unbalanced and with many data points, null registers will be dropped.')




#dropping NaN values in order to convert
df_p = df.copy().dropna()
#creating adjusted field of saledate to get date on m-d-Y format
df_p['saledate_adj'] = df_p['saledate'].apply(lambda x: datetime.strptime(converte_mes(x[4:15]),"%m %d %Y")  if isinstance(x, str) else x)
#creating converted date field
df_p['conv_data'] = df_p['saledate_adj'].map(datetime.toordinal)
#creating year field as integer
df_p['saleyear'] = df_p['saledate_adj'].apply(lambda x: int(x.year))

df_p = df_p.drop(columns = ['saledate_adj','conv_data'])
###############################################################################################################################


######################################## Validity Section #####################################################################

with st.expander('Validity'):
    st.header('Validity')
    st.subheader('Are columns with expected values? Are they balanced? How does the data distributes?')

    option = st.selectbox(
    "Select which column to inspect",
        list(df_p.columns)
    )

    df_p1 = df_p[[option]]

    df_p2 = df_p.copy()
    df_p2['make_adjusted'] = df_p['make'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['model_adjusted'] = df_p['model'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['trim_adjusted'] = df_p['trim'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['body_adjusted'] = df_p['body'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['transmission_adjusted'] = df_p['transmission'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    #df_p2['vin_adjusted'] = df_p['vin'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['state_adjusted'] = df_p['state'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['color_adjusted'] = df_p['color'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['interior_adjusted'] = df_p['interior'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['seller_adjusted'] = df_p['seller'].apply(lambda x : x.lower()  if isinstance(x, str) else x)


    column_name = df_p1.columns
    column_type = df_p1[column_name[0]].dtype 

    st.write(column_type)

    if column_type == 'object':
        st.write('Variável Categórica')

        #Numero de rótulos e wordcloud sem tratamento
        nunique_labels = len(df_p1.groupby(column_name[0]).count().index)
        labels = df_p1.groupby(column_name[0]).count().index
        text = str(df_p1[column_name[0]].values)
        wordcloud = WordCloud(max_font_size = 50, max_words = 999999, background_color= 'white').generate(text)

        st.write('Number of Labels: ', nunique_labels)
        st.write('Wordcloud of Labels ')

        fig, ax = plt.subplots(figsize = (10,8))
        ax.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(fig)

        #Numero de rótulos e wordcloud com tratamento de lower
        adjusted_column_name = str(column_name[0])
        nunique_labels = len(df_p2.groupby(adjusted_column_name).count().index)
        labels = df_p2.groupby(adjusted_column_name).count().index
        text = str(df_p2[adjusted_column_name].values)
        wordcloud = WordCloud(max_font_size = 50, max_words = 999999, background_color= 'white').generate(text)

        st.write('Number of Labels: ', nunique_labels)
        st.write('Wordcloud of Labels ')

        fig, ax = plt.subplots(figsize = (10,10))
        ax.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(fig)

        vec_i,vec_j, vec_similarity = [],[],[]

        for name_i in df_p2.groupby(adjusted_column_name).count().index:
            for name_j in df_p2.groupby(adjusted_column_name).count().index:
                vec_i.append(name_i)
                vec_j.append(name_j)
                vec_similarity.append(fuzz.ratio(name_i, name_j))
        #vetor de similaridade
        

        df_vec_i = pd.DataFrame(vec_i).rename(columns={0:'vec_i'})
        df_vec_j = pd.DataFrame(vec_j).rename(columns={0:'vec_j'})
        df_vec_similarity = pd.DataFrame(vec_similarity).rename(columns={0:'vec_similarity'})

        df_similarity = df_vec_i.merge(df_vec_j, left_index= True, right_index= True)
        df_similarity = df_similarity.merge(df_vec_similarity, left_index= True,right_index=True)

        df_similarity_plot = df_similarity.copy()
        df_similarity_plot = df_similarity_plot[df_similarity_plot['vec_similarity'] >= 75]
        df_similarity_plot = df_similarity_plot[df_similarity_plot['vec_similarity'] < 100]

        similarity_matrix = df_similarity_plot.pivot(index='vec_i', columns='vec_j',values='vec_similarity')
        heatmap = sns.heatmap(similarity_matrix)

        st.pyplot(heatmap.fig)

        del vec_i,vec_j,vec_similarity
        del df_vec_i,df_vec_j,df_vec_similarity
        del df_similarity_plot,similarity_matrix

    if column_type in ('float64','int64'):
        #writes which is the kind of variable 
        st.write('Numerical Variable')
        
        #writes next plot title
        st.write('Distribution')
        #plots the kde plot on seaborn
        kde = sns.displot(data = df_p1, x=column_name[0], kind='kde')
        fig, ax = plt.subplots(figsize = (10,8))
        st.pyplot(kde.figure)

        #writes the next plot title
        st.write('Q-Q plot')
        #q_q_plot = stats.probplot(df_p1[column_name[0]], dist="norm", plot=plt)
        #st.pyplot(q_q_plot)

        #plots de q-q plot based on altair
        base = alt.Chart(df_p1).transform_quantile(
                            column_name[0],
                            step=0.01,
                            #as_ = ['p', 'v']        
                        ).mark_point().encode(
                            x='prob:Q',
                            y='value:Q'
                        )
        st.altair_chart(base)

        #tests for lilliefors normality
        alpha = 0.05
        ksstat, pvalue = sm.stats.diagnostic.lilliefors(df_p1[column_name[0]].values)
        if pvalue > alpha:
            result = 'Normal'
        else:
            result = 'NOT Normal'
        #writes result on app
        st.write(f'Based on Lilliefors test, the Variable is:  {result:>21s}')

        st.write('Hexbin Plot X-Y')
    
        #treatment to get y column of sellingprice transformed on Box-Cox
        df_y = df_p[['sellingprice']].rename(columns={'sellingprice':'y'})
        #joining dfs
        df_hb = df_p1.merge(df_y, how='left', left_index= True, right_index= True)
        #hexbin plot with seaborn
        hb = sns.jointplot(data = df_hb, x=column_name[0], y = 'y' , kind="hex", color="#4CB391")
        fig, ax = plt.subplots(figsize = (10,8))
        st.pyplot(hb.figure)

###############################################################################################################################







######################################## Transformations Section ##############################################################

with st.expander('Transformation'):
    st.header('Transformation')
    st.subheader('Are columns with expected values? Are they balanced? How does the data distributes?')


    column_names = df_p.columns
    cont_columns = []
    for col in column_names: 
        if df_p[col].dtype != 'object':
            cont_columns.append(col)


    option_transf = st.selectbox(
    "Select which column to Transform",
        list(cont_columns)
    )

    df_p1 = df_p[[option_transf]]

    df_p2 = df_p.copy()
    df_p2['make_adjusted'] = df_p['make'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['model_adjusted'] = df_p['model'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['trim_adjusted'] = df_p['trim'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['body_adjusted'] = df_p['body'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['transmission_adjusted'] = df_p['transmission'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    #df_p2['vin_adjusted'] = df_p['vin'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['state_adjusted'] = df_p['state'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['color_adjusted'] = df_p['color'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['interior_adjusted'] = df_p['interior'].apply(lambda x : x.lower()  if isinstance(x, str) else x)
    df_p2['seller_adjusted'] = df_p['seller'].apply(lambda x : x.lower()  if isinstance(x, str) else x)

    column_name = df_p1.columns
    column_type = df_p1[column_name[0]].dtype 
    
    transformation_opt = st.radio(
        "Select one of the above transformations",
        ["Log","Box-Cox","Square Root",'Cubic Root']
    )

    normalization_opt = st.toggle('Choose to use RobustScaler() in the Sequence')
    
    def transformation(df_l,column,transformation_option,toggle):
        new_col = column + '_transformed'
        
        if transformation_option == 'Log':
            df_l[new_col] = df_l[column].apply(lambda x: np.log(x))

        if transformation_option == 'Box-Cox':
            boxcoxTr = PowerTransformer(method = "box-cox", standardize=True)
            b_r = boxcoxTr.fit_transform(df_l[column].values.reshape(-1,1))
            df_l[new_col] = b_r

        if transformation_option == 'Square Root':
            df_l[new_col] = df_l[column].apply(lambda x: np.sqrt(x))

        if transformation_option == 'Cubic Root':
            df_l[new_col] = df_l[column].apply(lambda x: np.cbrt(x))

        
        if toggle:
            st.write('Scaler Selected')
            transformer = RobustScaler()
            df_l[new_col] = transformer.fit_transform(df_l[new_col].values.reshape(-1,1))
        
        return df_l[new_col], new_col
    
    df_transf, col_transformed = transformation(df_p2,column_name[0],transformation_opt,normalization_opt)
    st.dataframe(df_transf)

    df_transf = pd.DataFrame(df_transf)
     #writes next plot title
    st.write('Distribution')
    #plots the kde plot on seaborn
    kde = sns.displot(data = df_transf, x=col_transformed, kind='kde')
    fig, ax = plt.subplots(figsize = (10,8))
    st.pyplot(kde.figure)

    #writes the next plot title
    st.write('Q-Q plot')
    #q_q_plot = stats.probplot(df_p1[column_name[0]], dist="norm", plot=plt)
    #st.pyplot(q_q_plot)

    #plots de q-q plot based on altair
    base = alt.Chart(df_transf).transform_quantile(
                            col_transformed,
                            step=0.01,
                            #as_ = ['p', 'v']        
                        ).mark_point().encode(
                            x='prob:Q',
                            y='value:Q'
                        )
    st.altair_chart(base)

    #tests for lilliefors normality
    alpha = 0.05
    ksstat, pvalue = sm.stats.diagnostic.lilliefors(df_transf[col_transformed].values)
    if pvalue > alpha:
        result = 'Normal'
    else:
        result = 'NOT Normal'
    #writes result on app
    st.write(f'Based on Lilliefors test, the Variable is:  {result:>21s}')


    st.write('Hexbin Plot X-Y')
    
    #treatment to get y column of sellingprice transformed on Box-Cox
    df_y_tranformed = df_p[['sellingprice']]
    boxcoxTr = PowerTransformer(method = "box-cox", standardize=True)
    b_r = boxcoxTr.fit_transform(df_y_tranformed['sellingprice'].values.reshape(-1,1))
    df_y_tranformed['sellingprice_transformed'] = b_r
    df_y_tranformed = df_y_tranformed.drop(columns = ['sellingprice']).rename(columns={'sellingprice_transformed':'y'})
    #joining dfs
    df_hb = df_transf.merge(df_y_tranformed, how='left', left_index= True, right_index= True)
    #hexbin plot with seaborn
    hb = sns.jointplot(data = df_hb, x=col_transformed, y = 'y' , kind="hex", color="#4CB391")
    fig, ax = plt.subplots(figsize = (10,8))
    st.pyplot(hb.figure)



#############################################################################################################################