"Analisis Statistik Regresi Linear"
# app.py
#!/home/nasri/anaconda3/envs/dashboard_env
# -*- coding: utf-8 -*-
from collections import OrderedDict
import streamlit as st
from streamlit.logger import get_logger # get_logger = class (?)
import functions_statistics #import function2 dr file demos_pertanian

LOGGER = get_logger(__name__) # constructor, properties class didefinisikan dalam method yaitu class atribute __name__

DEMOS = OrderedDict(  #membuat objek DEMOS dari class
    [
      ("Introduksi.", (functions_statistics.intro, None)),
        #(
        #    "Upload File.",
        #    (
        #        functions_statistics.open_file, None,
        #    ),
        ),
        (
           "1. Data.head() dan Statistik Diskripsi.",
            (
                functions_statistics.descriptive,
            ),           
        ),
        (
           "2. Korelasi.",
            (
                functions_statistics.korelasi,
            ),
        ),
        (
           "3. Visual Data.",
            (
                functions_statistics.visual_data,
            ),
        ),
        (
           "4. Regresi Multi Linear.",
            (
                functions_statistics.regresi,
            ),
        ),
        (
           "5. Regresi Linear (Parsial), Scatter Chart & Garis Regresi.",
            (
                functions_statistics.regresi_parsial,
            ),
        ),
        (
           "6. Evaluasi Model.",
            (
                functions_statistics.evaluasi,
            ),
        ),
        (
           "7. Uji Asumsi Regresi Linear dan Uji Validasi Model). ",
            (
                functions_statistics.uji_asumsi,
            ),
        ),
        (
           "8. Simulai Prediksi.",
            (
                functions_statistics.simulasi_produksi,
            ),
        ),      
    ]
)


def run():
    st.sidebar.markdown('<h1 style="color: white;"> author: m nasri aw </h1>', unsafe_allow_html=True)

    demo_name = st.sidebar.radio("Pilih Analisa Satatistik", list(DEMOS.keys()), 0)
    demo = DEMOS[demo_name][0]
    
    if demo_name == "Introduksi.":
        st.write("# Selamat Datang di Dashboard Analisis Statistik Regresi Linear.  👋")
        st.write("##### author: m nasri aw, email: nasri@stieimlg.ac.id; lecturer at https://www.stieimlg.ac.id/; Des 2024.")
        st.write(f"##### - Ketentuan: ")
        '''
        1. Input File: jenis file *.csv; Jika jenis spreadsheet (.*xlxs) di save as *.csv atau convert ke *.csv.
        2. Kolom-kolom di awal sebagai prediktor (Xi) dan kolom yang terakhir sebagai prediksi (Y), baris 1 untuk label kolom. Input File di load dari folder kerja maksimal 200 MB dan output dapat didownload ke penyimpan pengguna masing-masing. Tabel dan gambar output dapat didownload di masing-masing properties tabel atau gambar.
        3. Jenis data menggunakan angka atau bilangan integer, data skala lickert bisa digunakan dengan interpretasi sesuai skalanya. 
        4. Menggunakan regresi model Linear Regression dari libray sklearn.linear_model, statsmodels.api dan scipy.stats. Sebaiknya menggunakan data cukup besar (>1000 baris), 80 % untuk training model dan 20 % untuk uji model.
        5. Pemrograman menggunakan python dengan library streamlit dan library pengolah data, statistik dan library lainnya, source bersifat terbuka (opensource) yang dapat di download dari link github penulis. 
        6. Analisis statistik regresi meliputi:
           1. Diskripsi.
           2. Korelasi.
           3. Visual Data.
           4. Regresi Multi Linear.
           5. Regresi Linear (Parsial) & Scatter Chart Parsial x-y. 
           6. Evaluasi Model.
           7. Uji Asumsi Regresi Linear dan Uji Validasi Model.
           8. Simulasi Prediksi.
        7. Untuk link demo silahkan klik https://huggingface.co/spaces/nasriaw/regresi_linear; Selamat belajar semoga memudahkan untuk memahami statistik regresi.
        '''
    else:
#        show_code = st.sidebar.checkbox("Show code", True)
        st.markdown("# %s" % demo_name)
    demo()

#     if show_code:
#         st.markdown("## Code")
#         sourcelines, _ = inspect.getsourcelines(demo)
#         st.code(textwrap.dedent("".join(sourcelines[1:])))


if __name__ == "__main__": #memanggil method class get_logger
    run() # menjalankan function run()
