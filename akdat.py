import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
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
    
    # Preprocessing data
    label_encoder = LabelEncoder()
    data['Heart Disease'] = label_encoder.fit_transform(data['Heart Disease'])

    # Memisahkan fitur (X) dan label (y)
    X = data.drop(columns=['Heart Disease'])
    y = data['Heart Disease']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan latih model Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Input variabel yang ingin dibandingkan dengan 'Heart Disease'
    comparison_var = st.selectbox(
        "Pilih variabel yang ingin dibandingkan dengan 'Heart Disease':",
        options=X.columns.tolist()  # List of columns in the dataset excluding 'Heart Disease'
    )

    # Menampilkan grafik jika variabel valid
    if comparison_var in data.columns:
        st.subheader(f"Perbandingan {comparison_var} Berdasarkan Penyakit Jantung")

        plt.figure(figsize=(12, 6))

        # Menghitung jumlah pasien berdasarkan variabel yang dipilih dan status Heart Disease
        aggregated_data = data.groupby([comparison_var, 'Heart Disease']).size().reset_index(name='count')

        # Menggunakan lineplot untuk menggambarkan hubungan antara variabel yang dipilih dan status 'Heart Disease'
        sns.lineplot(data=aggregated_data, x=comparison_var, y='count', hue='Heart Disease', marker='o')

        plt.title(f"Comparison of '{comparison_var}' Based on Heart Disease")
        plt.xlabel(comparison_var.capitalize())
        plt.ylabel(f"Jumlah Pasien Berdasarkan {comparison_var.capitalize()}")
        plt.legend(title="Heart Disease")

        plt.tight_layout()  # Untuk memastikan grafik tidak saling tumpang tindih
        st.pyplot(plt)

    # Prediction Form Visible by Default
    st.subheader("Formulir Prediksi Penyakit Jantung")
    
    # Input dari pengguna dengan pilihan (selectbox, slider, radio)
    age = st.slider("Umur:", 0, 100, 50)  # Range umur 0-100, default 50
    sex = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])  # Pilih jenis kelamin
    chest_pain = st.selectbox("Tipe Nyeri Dada", [1, 2, 3, 4])  # Pilih tipe nyeri dada (1-4)
    bp = st.slider("Tekanan Darah:", 50, 200, 120)  # Range tekanan darah 50-200, default 120
    cholesterol = st.slider("Kolesterol:", 100, 400, 200)  # Range kolesterol 100-400, default 200
    fbs_over_120 = st.radio("Gula Darah Puasa > 120", ["Tidak", "Ya"])  # Pilih gula darah puasa
    ekg_results = st.selectbox("Hasil EKG", [0, 1, 2])  # Pilih hasil EKG (0-2)
    max_hr = st.slider("Detak Jantung Maksimal:", 60, 200, 150)  # Range detak jantung 60-200, default 150
    exercise_angina = st.radio("Angina Saat Olahraga", ["Tidak", "Ya"])  # Pilih angina saat olahraga
    st_depression = st.slider("Depresi ST:", 0.0, 5.0, 1.0)  # Range depresi ST 0-5, default 1
    slope_of_st = st.selectbox("Kemiringan ST", [1, 2, 3])  # Pilih kemiringan ST (1-3)
    num_vessels = st.selectbox("Jumlah Pembuluh Darah Fluro", [0, 1, 2, 3])  # Pilih jumlah pembuluh darah
    thallium = st.selectbox("Hasil Tes Thallium", [3, 6, 7])  # Pilih hasil tes thallium

    # Mapping user input to numeric values
    sex = 1 if sex == "Laki-laki" else 0
    fbs_over_120 = 1 if fbs_over_120 == "Ya" else 0
    exercise_angina = 1 if exercise_angina == "Ya" else 0

    # Membuat DataFrame dari input pengguna
    user_data = pd.DataFrame([[age, sex, chest_pain, bp, cholesterol, fbs_over_120, 
                               ekg_results, max_hr, exercise_angina, st_depression, 
                               slope_of_st, num_vessels, thallium]], 
                             columns=X.columns)

    # Button to trigger prediction
    if st.button("Prediksi Penyakit Jantung"):
        # Prediksi menggunakan model
        prediction = model.predict(user_data)
        result = "Presence" if prediction[0] == 1 else "Absence"
        
        st.write(f"Prediksi: Penyakit Jantung {result}")

else:
    st.write("Silakan upload file CSV terlebih dahulu.")
