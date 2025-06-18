import streamlit as st
import pandas as pd
import joblib # Untuk memuat model

# --- Judul Aplikasi ---
st.title('Prediksi Kategori Waktu Lulus Mahasiswa')
st.write('Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus "Tepat Waktu" atau "Terlambat" berdasarkan beberapa faktor.')

# --- Memuat Model ---
@st.cache_data # Cache model agar tidak dimuat ulang setiap kali aplikasi berjalan
def load_model():
    try:
        model = joblib.load('model_graduation.pkl')
        return model
    except FileNotFoundError:
        st.error("File model 'model_graduation.pkl' tidak ditemukan. Pastikan model berada di direktori yang sama.")
        return None

nb_model = load_model()

if nb_model is not None:
    # --- Input Data Baru dari Pengguna ---
    st.header('Masukkan Data Mahasiswa:')

    col1, col2 = st.columns(2)

    with col1:
        new_ACT = st.number_input('Nilai ACT composite score:', min_value=0.0, max_value=36.0, value=25.0)
        new_GPA = st.number_input('Nilai rata-rata SMA (GPA):', min_value=0.0, max_value=4.0, value=3.0)
        new_education = st.number_input('Tingkat pendidikan orang tua (angka, misal 0-5):', min_value=0.0, value=3.0)

    with col2:
        new_SAT = st.number_input('Nilai SAT total score:', min_value=0.0, max_value=1600.0, value=1200.0)
        new_income = st.number_input('Pendapatan orang tua (dalam USD):', min_value=0.0, value=50000.0)

    # --- Tombol Prediksi ---
    if st.button('Prediksi Kategori Lulus'):
        try:
            # Buat DataFrame dari input baru
            new_data_df = pd.DataFrame(
                [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
                columns=['ACT composite score', 'SAT total score', 'high school gpa', 'parental income', 'parent_edu_numerical']
            )

            # Lakukan prediksi
            predicted_code = nb_model.predict(new_data_df)[0]

            # Konversi hasil prediksi ke label asli
            label_mapping = {1: 'Tepat Waktu', 0: 'Terlambat'}
            predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

            st.success(f"**Prediksi kategori masa studi adalah: {predicted_label}**")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
else:
    st.warning("Model belum dimuat. Mohon periksa kembali file model Anda.")

# --- Petunjuk Deployment ---
st.markdown(
    """
    ---
    ### **Cara Deployment ke Streamlit Cloud via GitHub:**

    1.  **Siapkan Repositori GitHub:**
        * Buat repositori baru di GitHub Anda (misal: `aplikasi-prediksi-kelulusan`).
        * Unggah file `app.py` dan `model_gradulation.pkl` ke repositori tersebut.
        * Pastikan Anda juga memiliki file `requirements.txt` yang berisi daftar pustaka yang digunakan:
            ```
            streamlit
            pandas
            scikit-learn # Jika model Anda dibuat dengan scikit-learn
            joblib
            ```

    2.  **Deployment di Streamlit Cloud:**
        * Kunjungi [Streamlit Cloud](https://share.streamlit.io/).
        * Login dengan akun GitHub Anda.
        * Klik "New app" atau "Deploy an app".
        * Pilih repositori GitHub tempat Anda menyimpan file aplikasi.
        * Atur "Main file path" ke `app.py` (atau nama file Python Anda).
        * Klik "Deploy!".

    Streamlit Cloud akan secara otomatis mengambil kode dari repositori GitHub Anda, menginstal dependensi dari `requirements.txt`, dan menjalankan aplikasi Anda.
    """
)
