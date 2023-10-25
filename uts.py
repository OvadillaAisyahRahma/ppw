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
upload_data, modeling, implementasi = st.tabs(["Upload Data", "Modeling", "Implementasi"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan dari PTA Trunojoyo Madura yang sudah dikelola dengan label (Komputasi dan RPL) dan dilakukan pemodelan. Dataset yang dihasilkan yaitu dataset LDA K-Means")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with modeling:
    st.write("""# Modeling""")
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
  
with implementasi:
    st.write("""# Implementasi""")
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Baca data
    data_x = pd.read_csv('https://raw.githubusercontent.com/Feb11F/dataset/main/Term%20Frequensi%20Berlabel%20Final.csv')
    data_x = data_x.dropna(subset=['Dokumen'])  # Menghapus baris yang memiliki NaN di kolom 'Dokumen'

    # Ubah kelas A menjadi 0 dan kelas B menjadi 1
    kelas_dataset_binary = [0 if kelas == 'RPL' else 1 for kelas in data_x['Label']]
    data_x['Label'] = kelas_dataset_binary

    # Bagi data menjadi data pelatihan dan data pengujian
    X = data_x['Dokumen']
    label = data_x['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

    # Vektorisasi teks menggunakan TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Latih model Naive Bayes
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    # Latih model LDA
    k = 3
    alpha = 0.1
    beta = 0.2
    lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
    proporsi_topik_dokumen = lda.fit_transform(X_train_tfidf)
        
    import pickle
    with open('lda.pkl', 'rb') as file:
        ldaa = pickle.load(file)
    with open('nb.pkl', 'rb') as file:
        bayes = pickle.load(file)
    with open('knn.pkl', 'rb') as file:
        knn = pickle.load(file)

    with st.form("my_form"):
        st.subheader("Implementasi")
        input_dokumen = st.text_input('Masukkan Judul Yang Akan Diklasfifikasi')
        input_vector = tfidf_vectorizer.transform([input_dokumen])
        submit = st.form_submit_button("submit")
        # Prediksi proporsi topik menggunakan model LDA
        proporsi_topik = lda.transform(input_vector)[0]
        if submit:
            st.subheader('Hasil Prediksi')
            inputs = np.array([input_dokumen])
            input_norm = np.array(inputs)
            input_pred = knn.predict(input_vector)[0]
        # Menampilkan hasil prediksi
            if input_pred==0:
                st.success('RPL')
                st.write(proporsi_topik)
            else  :
                st.success('KK')
                st.write(proporsi_topik)
