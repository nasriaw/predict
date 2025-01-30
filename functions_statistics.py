'''
Functions Analisa statistik
"Analisis Statistik Regresi Linear"
# functions_statistics.py
#!/home/nasri/anaconda3/envs/dashboard_env
# -*- coding: utf-8 -*-
'''
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
#from math import sqrt
import plotly.express as px
import statistics
from statistics import linear_regression

def intro():
    '''
    Selamat Datang
    '''

def open_file():
    csv = st.file_uploader("file ekstensi:  *.csv", type="csv")
    if csv is not None:
        df = pd.read_csv(csv)
        st.session_state["data"] = df
        
    if "data" in st.session_state:
        df = st.session_state["data"]

        #st.write(f"dimensi data: {df.shape}")
        #st.write("data head : ")
        #st.write(df.head()) 
    return df
df=open_file()
def descriptive():
    #df=open_file()
    #st.write(f"dimensi data: {df.shape}")
    #st.write("Data Head : ")
    #st.write(df.head())
    st.write("Data Diskripsi : ")
    st.write(df.describe())

def korelasi():
    df=open_file()
    st.write(f"dimensi data: {df.shape}")
    st.write(df.corr())
    
def visual_data():
    df=open_file()
    columns=df.columns
        #x = data.drop(data.columns[-1],axis=1)
        #y = data.iloc[:,-1:]
    for i in columns:
        st.write(f"##### Chart {(i)}")
        st.bar_chart(df[i]) #scatter, bar, line, area, altair

def regresi():
    df=open_file()
    st.write(f"dimensi data: {df.shape}")
    x = df.drop(df.columns[-1],axis=1)
    y = df.iloc[:,-1:]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    # menggunakan statsmodels OLS
    x = sm.add_constant(xtrain)
    model=sm.OLS(ytrain,x).fit()
    
    st.write("### 4.1 Regresi, menggunakan library: statsmodels Ordinary Least Squares (OLS). ")
    st.write(model.summary())

    st.write("### 4.2. Persamaan Regresi. ")
    columns=df.columns
    #st.write(model.params)
    #st.write(f"Persamaan Regresi, {columns[5]} = {model.params[0]:0.4f} + {model.params[1]:0.4f} {columns[0]} + {model.params[2]:0.4f} {columns[1]} + {model.params[3]:0.4f} {columns[2]} + {model.params[4]:0.4f} {columns[3]} + {model.params[5]:0.4f} {columns[5]}.")
    #atau dengan algoritma
    [m,n] =df.shape
    k=(n-1)
    st.write(f"Persamaan Regresi untuk Prediksi KepuasanKlient = Intercept {model.params[0]:0.04f} + Prediktor : ")
    for i in range(k):
        st.write(f"- Prediktor {(df.columns[i])} = {(model.params[i+1]):0.04f} x {(df.columns[i])}")
    st.write("Persamaan regresi ini akan digunakan untuk prediksi, simulasi prediksi ada di bagian akhir (bagian ke 8).")
    st.write("### 4.3. Uji Hipotesis Prediktor Parsial & Serentak : ")
    st.write("#### a. P-Value & Uji Hipotesis Prediktor Parsial: ")
    #st.write("Tabel p-value :")
    #st.write(model.pvalues)
    alpha = 0.05
    st.write(f"Alpha = {alpha}") 
    st.write(f"Variabel : {columns}")
    prediktor_x=columns[:-1]
    prediksi_y=columns[-1]
    for i in prediktor_x:
        st.write(f"p-value {i} = {model.pvalues[i]:0.4f}")
        if (model.pvalues[i]) > alpha :
            st.write(f'- Hipotesis H0: "Diduga faktor {i} Tidak Berpengaruh terhadap {prediksi_y}" : p-value= {model.pvalues[i]:0.4f} > {alpha}: Menerima H0: faktor {i} tidak ada pengaruh terhadap {prediksi_y}')
        else:
            st.write(f'- Hipotesis Ho: "Diduga faktor {i} Tidak Berpengaruh terhadap {prediksi_y}" : p-value= {model.pvalues[i]:0.4f} < {alpha}, Menolak H0: faktor {i} berpengaruh terhadap {prediksi_y}')

    st.write("#### b. F-value atau P-value & Uji Hipotesis Serentak: ")
    st.write(f'- F-value: {model.fvalue}; p-value: {model.f_pvalue}')
    st.write(f'- Prediktor X : {columns[:-1]}')
    st.write(f'- Prediksi Y : {columns[-1]}')
    st.write(f'Hipotesis H0: Diduga semua Prediktor X secara serentak tidak berpengaruh untuk memprediksi {prediksi_y}. ')
    if model.f_pvalue < 0.05 :
            st.write(f'- p-value = {model.f_pvalue:0.4f} < 0.05, Menolak H0: artinya semua Prediktor X secara serentak berpengaruh untuk memprediksi {prediksi_y}')
    else:
            st.write(f'- p-value = {model.f_pvalue:0.4f} > 0.05, Menerima H0: artinya semua Prediktor X secara serentak tidak berpengaruh untuk memprediksi {prediksi_y}')

def regresi_parsial():
    df=open_file()
    columns=df.columns
    [m,n] =df.shape
    k=n-1
    for i in range(k):
        df1=pd.DataFrame([df[columns[i]],df[columns[-1]]])        
        x =np.array(df1.iloc[0])
        y =np.array(df1.iloc[1])
        st.write(f"#### 5.{i+1}. Evaluation a model : {df.columns[i]} (x) - {df.columns[-1]} (y) = ")
        st.write(" Data Deskripsi = ")
        #st.write(df1.T.head(5))
        st.write(df1.T.describe())
        # menggunakan statsmodels OLS
        #x = sm.add_constant(x)
        #model_ols=sm.OLS(y,x).fit()
        model_ols = smf.ols("y ~ x", data=pd.DataFrame(x,y)).fit()
        #st.write(f" Parameter model : {model_ols.params}")  # cons (intercept) = {model_ols.params[0]:0.04f} ; coef = {model_ols.params[2]:0.04f}")
        st.write(model_ols.summary())
        
        st.write(f" - R-squared = {model_ols.rsquared:0.04f}")
        st.write(f" - MSE = {model_ols.mse_total:0.04f}")
        st.write(f" - Standard errors =  {(model_ols.bse)}")
        #st.write(f" - Predicted values =  {model_ols.predict()}")
               
        st.write(f"##### Chart: {df.columns[i]} - {df.columns[-1]}, dan Garis Regresi Linear = ")
        st.write(f" - Parameter model : intercept = {model_ols.params[0]:0.04f} ; coef = {model_ols.params[1]:0.04f} ")
        y1=model_ols.params[0] + (model_ols.params[1] * x)
        fig=px.scatter(
            y1, x,
            y, x,
            trendline ="ols")
        st.plotly_chart(fig)

def evaluasi():
    df=open_file()
    x = df.drop(df.columns[-1],axis=1)
    y = df.iloc[:,-1:]
    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    #menggunakan sklearn.linear_model
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    
    st.write("##### Evaluasi Model Regresi: MSE, MAE, R_square, menggunakan library: sklearn.linear_model")
    pred = model.predict(xtest)
    mse = mean_squared_error(ytest,pred)
    st.write(f"- MSE: {mse:0.4f}")
    rmse=mean_squared_error(ytest, pred, squared=False)
    st.write(f"-  RMSE: {rmse:0.4f}")
    mae = mean_absolute_error(ytest, pred)
    st.write(f"-  MAE: {mae:0.4f}")
    # R_square
    from sklearn.metrics import r2_score
    r2 = r2_score(ytest, pred)
    st.write(f'- R square : {r2:0.4f}')
        
def uji_asumsi():
    df=open_file()
    x = df.drop(df.columns[-1],axis=1)
    y = df.iloc[:,-1:]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
        
    st.write("#### 7.1. Cek Normalitas, menggunakan shapiro test, library: scipy.stats")
    st.write("##### -  Hipotesis H0: Diduga sample berdistribusi NORMAL")
    st.write("##### -  Hipotesis HA: Diduga sample berdistribusi TIDAK NORMAL")
    alpha=0.05
    columns=df.columns
    for i in columns:
        st.write(f"##### Distribusi {(i)}")
        a,b=stats.shapiro(df[[i]])
        st.write(f"- Statistics, {a:0.04f}, p-value, {b:0.04f}")
        if b < alpha:
            st.write(f"- p-value {b:0.04f} < {alpha}: H0 ditolak, {i} berdistribusi TIDAK NORMAL")
        else:
            st.write(f"- p-value {b:0.04f} > {alpha}: H0 diterima, {i} berdistribusi NORMAL")

    st.write("#### 7.2. Alternatif Cek Normalitas, menggunakan Kolmogorov-Smirnov test, library: scipy.stats")
    st.write("##### - Hipotesis H0: Diduga sample berdistribusi NORMAL")
    st.write("##### - Hipotesis HA: Diduga sample berdistribusi TIDAK NORMAL")
    for i in columns:
        st.write(f"##### Distribusi {(i)}")
        a,b=stats.kstest(df[[i]], 'norm')
        st.write(f"- Statistics, {a}, p-value, {b}")
        if b < alpha:
            st.write(f"- p-value {b} < {alpha}: H0 ditolak, {i} berdistribusi TIDAK NORMAL")
        else:
            st.write(f"- p-value {b} > {alpha}: H0 diterima, {i} berdistribusi NORMAL")

    st.write("#### 7.3 Chart y-test dan y-predict")
    st.write("##### a. Chart y-test : ")
    st.scatter_chart(ytest)
    st.write("##### b. Chart y-predict : ")
    st.scatter_chart(pred)

    st.write("#### 7.4.Cek Multicolinearitas dan Homoskesdasitas")
    st.write(f"##### 7.4.a. Cek Multicolinearity, menggunakan library: sklearn.metrics")
    from sklearn.metrics import r2_score
    r2 = r2_score(ytest, pred)
    VIF = 1/(1- r2)
    st.write(f'- R square : {r2:0.4f}')
    if VIF < 5 :
        st.write(f'- VIF: {VIF:0.4f} < 5 : Tidak terjadi Multicolinearity')
    else:
        st.write(f'-  VIF: {VIF:0.4f} > 5 : Terjadi Multicolinearity')

    st.write(f"##### 7.4.b. Cek Homoskesdasitas, menggunakan library: scipy.stats")
    x = df.drop(df.columns[-1],axis=1)
    y = df.iloc[:,-1:]

    rho, p_value = stats.spearmanr(x.iloc[:,0:1], y)
    label_x=columns[:-1]
    label_y=columns[-1]
    x = df.drop(df.columns[-1],axis=1)
    y = df.iloc[:,-1:]
    for i in x:
        rho, p_value = stats.spearmanr(x[i], y)
        st.write(f"Spearman correlation coefficient {i} - {label_y} :  rho = {rho:0.04f}; & p-value =  {p_value:0.04f}")
        if (p_value) < 0.05 :
            st.write(f'- p-value: {p_value:0.04f} < 0.05 : Homokesdasitas atau Heterokesdasitas tidak terjadi')
        else:
            st.write(f'- p-value: {p_value:0.04f} > 0.05 : Terjadi Homokesdasitas atau Heterokesdasitas')

    st.write(f"#### 7.5. Test Validitas Instrumen Penelitian, menggunakan library: sklearn.metrics")
    r2 = r2_score(ytest, pred)
    st.write(f'- R square : {r2:0.4f}')
    if r2 > 0.3 :
        st.write(f'- R-square: {r2:0.04f} > 0.3 : instrumen penelitian dinyatakan VALID ')
    else:
        st.write(f'- R-square: {r2:0.04f} < 0.3 : instrumen penelitian dinyatakan TIDAK VALID ')

    st.write(f"#### 7.6. Test Validitas Instrumen Penelitian, menggunakan library: pingouin")
    cronbach=(f'{pg.cronbach_alpha(data=df)[0]:0.04f}')
    st.write(f'- cronbach alpha average  : {cronbach}')
    if cronbach >= str(0.9) :
        st.write(f'- cronbach alpha average: {cronbach} > 0.9 : Internal consistensi instrumen penelitian dinyatakan: Excellent')
    elif cronbach >= str(0.8) :
        st.write(f'- cronbach alpha average: {cronbach} > 0.8 : Internal consistensi instrumen penelitian dinyatakan: Good')
    elif cronbach >= str(0.7) :
        st.write(f'- cronbach alpha average: {cronbach} > 0.7 : Internal consistensi instrumen penelitian dinyatakan: Acceptable')
    elif cronbach >= str(0.6) :
        st.write(f'- cronbach alpha average: {cronbach} > 0.6 : Internal consistensi instrumen penelitian dinyatakan: Questionable')
    elif cronbach >= str(0.5) :
        st.write(f'- cronbach alpha average: {cronbach} > 0.5 : Internal consistensi instrumen penelitian dinyatakan: Poor')    
    else:
        st.write(f'- cronbach alpha average: {cronbach} < 0.5 : Internal consistensi instrumen penelitian dinyatakan: Unacceptable')
    
def simulasi_produksi():
    df=open_file()
    x = df.drop(df.columns[-1],axis=1)
    y = df.iloc[:,-1:]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    x = sm.add_constant(xtrain)
    model=sm.OLS(ytrain,x).fit()
    #st.write("Persamaan Regresi. ")
    columns=df.columns
    prediktor_x=columns[:-1]
    prediksi_y=columns[-1]

    [m,n] =df.shape
    k=(n-1)
    st.write(f"dimensi prediktor: {k}")
    st.write("##### Masukkan nilai prediktor: ") 
    
    output=[]
    for i in range(k):    
        feature=st.number_input(f'##### {prediktor_x[i]},  input range:  {df[columns[i]].min()} - {df[columns[i]].max()}', value=df[columns[i]].mean())
        output.append(feature)
        df1 = (pd.DataFrame(output, columns=([''])))
    st.write(f"parameter regresi = {model.params}")
    [k,l]=df1.shape
    sum_prediktor=0
    for i in range(k):
        sum_prediktor += (model.params[i+1])*(df1.iloc[i])
    #st.write(f" jumlah prediktor: {sum_prediktor}")
    predik2=(model.params)[0] + sum_prediktor
    st.write(f"#### Prediksi {prediksi_y} = {predik2}")

