import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Description
st.title("Heart Disease Prediction App")
st.write("This app predicts the presence of heart disease based on user input and visualizes data analysis.")

# Upload the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    
    # Menampilkan data awal
    st.write("Data Awal:", data.head())

    # Label Encoding untuk kolom kategori (contoh: Heart Disease)
    st.subheader("Preprocessing Data")
    label_encoder = LabelEncoder()
    if 'Heart Disease' in data.columns:
        data['Heart Disease'] = label_encoder.fit_transform(data['Heart Disease'])
    
    # Pilihan untuk Normalisasi
    st.write("Pilih kolom untuk normalisasi:")
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    cols_to_normalize = st.multiselect("Kolom Fitur Numerik:", numeric_cols)

    # Normalisasi hanya jika ada kolom yang dipilih
    if cols_to_normalize:
        st.write(f"Melakukan normalisasi pada kolom: {cols_to_normalize}")
        scaler = MinMaxScaler()
        data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])
        st.write("Data setelah normalisasi:", data.head())
    else:
        st.write("Tidak ada kolom yang dipilih untuk normalisasi.")

    # Pisahkan fitur (X) dan label (y)
    X = data.drop(columns=['Heart Disease'])
    y = data['Heart Disease']

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan latih model Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Visualisasi Data
    comparison_var = st.selectbox("Pilih variabel yang ingin dibandingkan dengan 'Heart Disease':", options=X.columns.tolist())
    chart_type = st.selectbox("Pilih tipe grafik:", ["Line Chart", "Histogram"])

    if comparison_var in data.columns:
        st.subheader(f"Perbandingan {comparison_var} Berdasarkan Penyakit Jantung")

        # Visualisasi data berdasarkan pilihan grafik
        plt.figure(figsize=(12, 6))
        
        if chart_type == "Line Chart":
            aggregated_data = data.groupby([comparison_var, 'Heart Disease']).size().reset_index(name='count')
            sns.lineplot(data=aggregated_data, x=comparison_var, y='count', hue='Heart Disease', marker='o')
            plt.title(f"Comparison of '{comparison_var}' Based on Heart Disease")
            plt.xlabel(comparison_var.capitalize())
            plt.ylabel(f"Jumlah Pasien Berdasarkan {comparison_var.capitalize()}")
            plt.legend(title="Heart Disease")
        
        elif chart_type == "Histogram":
            sns.histplot(data=data, x=comparison_var, hue="Heart Disease", multiple="stack", bins=20)
            plt.title(f"Distribution of '{comparison_var}' Based on Heart Disease")
            plt.xlabel(comparison_var.capitalize())
            plt.ylabel("Count")
            plt.legend(title="Heart Disease")
        
        plt.tight_layout()
        st.pyplot(plt)

    # Formulir Prediksi
    st.subheader("Formulir Prediksi Penyakit Jantung")
    # Input dari pengguna
    age = st.slider("Umur:", 0, 100, 50)
    sex = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    chest_pain = st.selectbox("Tipe Nyeri Dada", [1, 2, 3, 4])
    bp = st.slider("Tekanan Darah:", 50, 200, 120)
    cholesterol = st.slider("Kolesterol:", 100, 400, 200)
    fbs_over_120 = st.radio("Gula Darah Puasa > 120", ["Tidak", "Ya"])
    ekg_results = st.selectbox("Hasil EKG", [0, 1, 2])
    max_hr = st.slider("Detak Jantung Maksimal:", 60, 200, 150)
    exercise_angina = st.radio("Angina Saat Olahraga", ["Tidak", "Ya"])
    st_depression = st.slider("Depresi ST:", 0.0, 5.0, 1.0)
    slope_of_st = st.selectbox("Kemiringan ST", [1, 2, 3])
    num_vessels = st.selectbox("Jumlah Pembuluh Darah Fluro", [0, 1, 2, 3])
    thallium = st.selectbox("Hasil Tes Thallium", [3, 6, 7])

    # Mapping input ke nilai numerik
    sex = 1 if sex == "Laki-laki" else 0
    fbs_over_120 = 1 if fbs_over_120 == "Ya" else 0
    exercise_angina = 1 if exercise_angina == "Ya" else 0

    # DataFrame dari input pengguna
    user_data = pd.DataFrame([[age, sex, chest_pain, bp, cholesterol, fbs_over_120, 
                               ekg_results, max_hr, exercise_angina, st_depression, 
                               slope_of_st, num_vessels, thallium]], 
                             columns=X.columns)

    # Prediksi
    if st.button("Prediksi Penyakit Jantung"):
        prediction = model.predict(user_data)
        result = "Presence" if prediction[0] == 1 else "Absence"
        st.write(f"Prediksi: Penyakit Jantung {result}")

else:
    st.write("Silakan upload file CSV terlebih dahulu.")
