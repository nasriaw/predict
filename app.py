"Analisis Statistik Regresi Linear"
# app.py
#!/home/nasri/anaconda3/envs/dashboard_env
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import scipy.stats as stats
import pingouin as pg

st.write("## Selamat Datang di Dashboard Analisis Statistik Regresi Linear.")
st.write("##### author: m nasri aw, email: nasriaw@gmail.com; Des 2024.")
st.write(f"##### - Ketentuan: ")
'''
1. Input File: *.csv; dengan kolom di awal sebagai prediktor (X) dan kolom terakhir sebagai prediksi (Y), baris 1 untuk label kolom. Input File di load dari folder kerja maksimal 200 MB dan output dapat didownload ke penyimpan pengguna masing-masing,Tabel dan gambar output dapat didownload di masing-masing properties tabel atau gambar.
2.Jenis data menggunakan angka atau bilangan integer, data skala lickert bisa digunakan dengan interpretasi sesuai skalanya. 
3. Menggunakan regresi model Linear Regression dari libray sklearn.linear_model, statsmodels.api dan scipy.stats. Sebaiknya menggunakan data cukup besar (>1000 baris), 80 % untuk training model dan 20 % untuk uji model.
4. Pemrograman menggunakan python dengan library dashboard streamlit dan library pengolah data, statistik dan library lainnya, source bersifat terbuka (opensource) yang dapat di download dari link github penulis. 
5. Analisis statistik regresi meliputi:
   1. Diskripsi.
   2. Korelasi.
   3. Visual Data.
   4. Regresi.
   5. Evaluasi Model.
   6. Uji Asumsi Regresi Linear dan Uji Validasi Model.
   7. Simulasi Prediksi.
6. Selamat belajar semoga memudahkan untuk memahami statistik regresi.
'''

csv = st.file_uploader("#### Upload a file :", type="csv")

if csv is not None:
    df = pd.read_csv(csv)
    st.session_state["data"] = df

if "data" in st.session_state:
    df = st.session_state["data"]

    st.write(f"dimensi data: {df.shape}")
    st.write("data head : ")
    st.write(df.head())

    st.write("### 1. Diskripsi Data:")
    st.write(df.describe(include='all').fillna("").astype("str"))
    st.write("### 2. Korelasi Data:")
    st.write(df.corr())
    st.write("### 3. Visual Data:")
    columns=df.columns
        #x = data.drop(data.columns[-1],axis=1)
        #y = data.iloc[:,-1:]
    for i in columns:
        st.write(f"##### Chart {(i)}")
        st.line_chart(df[i]) #scatter, bar, line, area, altair

    st.write("### 4. Regresi : ")
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
    st.write("Persamaan regresi ini akan digunakan untuk prediksi, simulasi prediksi ada di bagian akhir (bagian ke 7).")
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

    st.write("### 5. Evaluasi Model : ")
    model = LinearRegression()
    model.fit(xtrain, ytrain)

    st.write("##### Evaluasi Model Regresi: MSE, MAE, R_square, menggunakan library: sklearn.linear_model")
    pred = model.predict(xtest)
    mse = mean_squared_error(ytest,pred)
    st.write(f"- MSE: {mse:0.4f}")
    #rmse=mean_squared_error(ytest, pred, squared=False)
    #st.write(f"- RMSE: {rmse:0.4f}")
    mae = mean_absolute_error(ytest, pred)
    st.write(f"- MAE: {mae:0.4f}")
    # R_square
    from sklearn.metrics import r2_score
    r2 = r2_score(ytest, pred)
    st.write(f'- R square : {r2:0.4f}')

    st.write("### 6. Uji Asumsi Regresi Linear dan Uji Validasi Model")
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)

    st.write("#### 6.1.a.Cek Normalitas, dengan Chart melihat relasi linear nya")
    columns=df.columns
    for i in columns:
        st.write(f"##### Chart {(i)}")
        st.scatter_chart(df[i])
        
    st.write("#### 6.1.b.Cek Normalitas, menggunakan shapiro test, library: scipy.stats")
    st.write("- Hipotesis H0: Diduga sample berdistribusi NORMAL")
    st.write("- Hipotesis HA: Diduga sample berdistribusi TIDAK NORMAL")
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

    st.write("#### 6.1.c. Alternatif Cek Normalitas, menggunakan Kolmogorov-Smirnov test, library: scipy.stats")
    st.write("- Hipotesis H0: Diduga sample berdistribusi NORMAL")
    st.write("- Hipotesis HA: Diduga sample berdistribusi TIDAK NORMAL")
    for i in columns:
        st.write(f"##### Distribusi {(i)}")
        a,b=stats.kstest(df[[i]], 'norm')
        st.write(f"- Statistics, {a}, p-value, {b}")
        if b < alpha:
            st.write(f"- p-value {b} < {alpha}: H0 ditolak, {i} berdistribusi TIDAK NORMAL")
        else:
            st.write(f"- p-value {b} > {alpha}: H0 diterima, {i} berdistribusi NORMAL")

    st.write("#### 6.2. Chart y-test dan y-predict")
    st.write("##### a. Chart y-test : ")
    st.scatter_chart(ytest)
    st.write("##### b. Chart y-predict : ")
    st.scatter_chart(pred)

    st.write("#### 6.3.Cek Multicolinearitas dan Homoskesdasitas")
    st.write(f"##### 6.3.a. Cek Multicolinearity, menggunakan library: sklearn.metrics")
    from sklearn.metrics import r2_score
    r2 = r2_score(ytest, pred)
    VIF = 1/(1- r2)
    st.write(f'- R square : {r2:0.4f}')
    if VIF < 5 :
        st.write(f'- VIF: {VIF:0.4f} < 5 : Tidak terjadi Multicolinearity')
    else:
        st.write(f'-  VIF: {VIF:0.4f} > 5 : Terjadi Multicolinearity')

    st.write(f"##### 6.3.b. Cek Homoskesdasitas, menggunakan library: scipy.stats")
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

    st.write(f"#### 6.4. Test Validitas Instrumen Penelitian, menggunakan library: sklearn.metrics")
    r2 = r2_score(ytest, pred)
    st.write(f'- R square : {r2:0.4f}')
    if r2 > 0.3 :
        st.write(f'- R-square: {r2:0.04f} > 0.3 : instrumen penelitian dinyatakan VALID ')
    else:
        st.write(f'- R-square: {r2:0.04f} < 0.3 : instrumen penelitian dinyatakan TIDAK VALID ')

    st.write(f"#### 6.5. Test Validitas Instrumen Penelitian, menggunakan library: pingouin")
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
     
    st.write("### 7. Simulai Prediksi : ")
    # menggunakan statsmodels OLS
    x = sm.add_constant(xtrain)
    model=sm.OLS(ytrain,x).fit()

    #st.write("Persamaan Regresi. ")
    columns=df.columns
    #st.write(model.params)
    #st.write(f"Prediksi: {columns[5]} = {model.params[0]:0.4f} + {model.params[1]:0.4f} {columns[0]} + {model.params[2]:0.4f} {columns[1]} + {model.params[3]:0.4f} {columns[2]} + {model.params[4]:0.4f} {columns[3]} + {model.params[5]:0.4f} {columns[4]}")   

    columns=df.columns
    prediktor_x=columns[:-1]
    prediksi_y=columns[-1]

    [m,n] =df.shape
    k=(n-1)
    st.write(f"dimensi prediktor: {k}")
    st.write("##### Masukkan nilai prediktor: ") 
    
    output=[]
    for i in range(k):    
        feature=st.number_input(f'{prediktor_x[i]},  input range:  {df[columns[i]].min()} - {df[columns[i]].max()}', value=df[columns[i]].mean())
        output.append(feature)
        df1 = (pd.DataFrame(output, columns=([''])))
    predik=model.params[0] + model.params[1]*df1.iloc[0] + model.params[2]*df1.iloc[1]+ model.params[3]*df1.iloc[2]+ model.params[4]*df1.iloc[3]+ model.params[5]*df1.iloc[4]
    #st.write(f"#### Prediksi {columns[5]} : {float(predik):0.04f}")

    st.write(f"parameter regresi = {model.params}")
    [k,l]=df1.shape
    sum_prediktor=0
    for i in range(k):
        sum_prediktor += (model.params[i+1])*(df1.iloc[i])
    #st.write(f" jumlah prediktor: {sum_prediktor}")
    predik2=(model.params)[0] + sum_prediktor
    st.write(f"#### Prediksi {prediksi_y} = {predik2}")

