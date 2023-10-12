import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


st.title("UTS PPW KELAS A (MODELLING)")
st.write("-------------------------------------------------------------------------------------------------------------------------")
st.write("**Nama  : Ovadilla Aisyah Rahma**")
st.write("**NIM   : 200411100033**")
st.write("-------------------------------------------------------------------------------------------------------------------------")
upload_data, modeling = st.tabs(["Upload Data", "Modeling"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan dari PTA Trunojoyo Madura yang sudah dikelola dengan label (Komputasi dan RPL) dan dilakukan pemodelan. Dataset yang dihasilkan yaitu dataset LDA K-Means")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

#with preporcessing :
    #st.write("""# Preprocessing""")
    #st.subheader("""Normalisasi Data""")
    #st.markdown("""
    #Dimana :
    #- X = data yang akan dinormalisasi atau data asli
    #- min = nilai minimum semua data asli
    #- max = nilai maksimum semua data asli
    #""")
    #df = df.drop(columns=['Dokumen'])
    #Mendefinisikan Varible X dan Y
    #X = df.drop(columns=['Label'])
    #y = df['Cluster'].values
    #df
    #X
    #df_min = X.min()
    #df_max = X.max()
    
    #NORMALISASI NILAI X
    #scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    #scaled = scaler.fit_transform(X)
    #features_names = X.columns.copy()
    #features_names.remove('label')
    #scaled_features = pd.DataFrame(scaled, columns=features_names)

    #st.subheader('Hasil Normalisasi Data')
    #st.write(scaled_features)

    #st.subheader('Target Label')
    #dumies = pd.get_dummies(df.Cluster).columns.values.tolist()
    #dumies = np.array(dumies)

    #labels = pd.get_dummies(df.Cluster).columns.values.tolist()

    #st.write(labels)

with modeling:
    st.write("""# Modeling""")
#    training, test = train_test_split(scaled_features,test_size=0.2, random_state=42)#Nilai X training dan Nilai X testing
#    training_label, test_label = train_test_split(y, test_size=0.2, random_state=42)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        #NaiveBayes
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Data pelatihan
        X_train = df[['Topik 1', 'Topik 2', 'Topik 3', 'Topik 4', 'Topik 5', 'Topik 6']]
        y_train = df['Cluster']
        
        # Bagi data menjadi data pelatihan dan data pengujian
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Inisialisasi model Naive Bayes
        naive_bayes = GaussianNB()
        
        # Latih model Naive Bayes
        naive_bayes.fit(X_train, y_train)
        
        # Prediksi kluster data pengujian
        y_pred = naive_bayes.predict(X_test)
        
        # Hitung akurasi
        accuracy_gausian = accuracy_score(y_test, y_pred)
        
        #print("Akurasi Naive Bayes:", accurac)
                
        #KNN
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Data contoh: 'Cluster' adalah label yang berasal dari kelompok K-Means
        X = df[['Topik 1', 'Topik 2', 'Topik 3', 'Topik 4', 'Topik 5']]
        y = df['Cluster']
        
        # Bagi data menjadi data pelatihan dan data pengujian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Inisialisasi model K-Nearest Neighbors (KNN) dengan n_neighbors=3
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        
        # Latih model dengan data pelatihan
        knn_classifier.fit(X_train, y_train)
        
        # Lakukan prediksi menggunakan data pengujian
        y_pred = knn_classifier.predict(X_test)
        
        # Evaluasi model
        accuracy_knn = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Tampilkan hasil
        #print("Akurasi: {:.2f}%".format(accuracy * 100))
        #print("\nLaporan Klasifikasi:\n", report)


        #Decission Tree
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        X = df[['Topik 1', 'Topik 2', 'Topik 3', 'Topik 4', 'Topik 5', 'Topik 6']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, df['Cluster'], test_size=0.2, random_state=42)
        
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, y_train)
        
        y_pred = decision_tree.predict(X_test)
        
        accuracy_DT = accuracy_score(y_test, y_pred)
        #print("Akurasi model Decision Tree:", accuracy)
        
        #print("Laporan Klasifikasi:")
        #print(classification_report(y_test, y_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(accuracy_gausian))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(accuracy_knn))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(accuracy_DT))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [accuracy_gausian,accuracy_knn, accuracy_DT],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  
