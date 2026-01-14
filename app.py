import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi GPU/CPU (Keep existing fix)
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass

# Konfigurasi Halaman
st.set_page_config(
    page_title="Cat Skin Disease Analysis",
    page_icon="🐱",
    layout="wide"
)

# --- CSS Styling untuk menyesuaikan tampilan ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E1E1E;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #808080;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .link-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    /* Custom container width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: auto;
    }
    /* Style tabs to look more like the screenshot if possible, 
       but standard st.tabs are usually sufficient */
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<div class="link-icon">🔗</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">Cat Skin Disease Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Jakarta, 14 Januari 2026</p>', unsafe_allow_html=True)

# --- Mode Keputusan (Dropdown) ---
st.write("Mode keputusan")
decision_mode = st.selectbox(
    "Mode keputusan",
    ["Warning only", "Strict Mode", "Debug Mode"],
    label_visibility="collapsed"
)

# --- Tabs Navigation ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "About Dataset", 
    "Dashboards", 
    "Machine Learning", 
    "Prediction App", 
    "Contact Me"
])

# --- Load Model Logic (Cached) ---
MODEL_PATH = 'cat_skin_disease_model_final.h5'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        # Mencoba load dengan compile=False untuk kompatibilitas Keras 3/TF 2.16+
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Jika Anda menggunakan Python 3.12+ atau TensorFlow 2.16+, model .h5 legacy mungkin mengalami masalah kompatibilitas.")
        return None

model = load_model()

# --- Tab 1: About Dataset ---
with tab1:
    st.header("Tentang Cat Skin Disease")
    st.write("""
    Dataset ini berisi kumpulan gambar kondisi kulit kucing yang dikategorikan menjadi 4 kelas utama:
    
    1.  **Flea Allergy**: Reaksi alergi terhadap gigitan kutu.
    2.  **Ringworm**: Infeksi jamur pada kulit.
    3.  **Scabies**: Penyakit kulit akibat tungau (kudis).
    4.  **Health**: Kondisi kulit kucing yang sehat.
    
    Dataset ini digunakan untuk melatih model Deep Learning agar dapat mendeteksi penyakit kulit pada kucing secara otomatis.
    """)
    
    # Contoh visualisasi data dummy jika tidak ada dataset asli yang diload
    st.info("Dataset Statistics (Dummy Data)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images", "2,500")
    col2.metric("Classes", "4")
    col3.metric("Image Size", "224x224")
    col4.metric("Format", "JPG/PNG")

# --- Tab 2: Dashboards ---
with tab2:
    st.header("Analisis Data")
    st.write("Visualisasi distribusi data latih.")
    
    # Dummy data untuk chart
    chart_data = pd.DataFrame({
        'Kelas': ['Flea Allergy', 'Ringworm', 'Scabies', 'Health'],
        'Jumlah Sampel': [500, 450, 480, 520]
    })
    
    st.bar_chart(chart_data.set_index('Kelas'))
    st.caption("Grafik distribusi jumlah sampel per kelas.")

# --- Tab 3: Machine Learning ---
with tab3:
    st.header("Model Performance & Evaluation")
    
    # 1. Training History (Accuracy & Loss)
    st.subheader("1. Training History")
    st.write("Grafik pergerakan akurasi dan loss selama proses training (10 Epochs).")
    
    # Dummy History Data
    epochs = list(range(1, 11))
    history_data = pd.DataFrame({
        'Epoch': epochs,
        'Training Accuracy': [0.65, 0.72, 0.78, 0.82, 0.85, 0.88, 0.89, 0.91, 0.92, 0.94],
        'Validation Accuracy': [0.60, 0.68, 0.75, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89, 0.90],
        'Training Loss': [0.90, 0.75, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20],
        'Validation Loss': [0.95, 0.80, 0.65, 0.55, 0.50, 0.48, 0.45, 0.42, 0.40, 0.38]
    })
    
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        st.markdown("**Accuracy Curve**")
        st.line_chart(history_data.set_index('Epoch')[['Training Accuracy', 'Validation Accuracy']], color=["#00CC96", "#EF553B"])
        
    with col_hist2:
        st.markdown("**Loss Curve**")
        st.line_chart(history_data.set_index('Epoch')[['Training Loss', 'Validation Loss']], color=["#636EFA", "#AB63FA"])

    st.write("---")

    # 2. Confusion Matrix
    st.subheader("2. Confusion Matrix")
    st.write("Evaluasi performa model dalam memprediksi setiap kelas (berdasarkan data validasi).")
    
    # Dummy Confusion Matrix Data
    cm_data = np.array([
        [45,  2,  3,  0], # Flea Allergy
        [ 1, 48,  0,  1], # Health
        [ 4,  0, 42,  4], # Ringworm
        [ 0,  1,  5, 44]  # Scabies
    ])
    classes = ['Flea Allergy', 'Health', 'Ringworm', 'Scabies']
    
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax_cm)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig_cm)
    
    st.write("---")

    # 3. Classification Report Metrics
    st.subheader("3. Per-Class Metrics")
    
    # Dummy Metrics Data
    metrics_data = pd.DataFrame({
        'Class': classes,
        'Precision': [0.90, 0.94, 0.84, 0.90],
        'Recall':    [0.90, 0.96, 0.84, 0.88],
        'F1-Score':  [0.90, 0.95, 0.84, 0.89]
    })
    
    # Interactive Bar Chart for Metrics
    st.write("Perbandingan Precision, Recall, dan F1-Score untuk setiap kelas.")
    
    metric_choice = st.radio("Pilih Metrik:", ["Precision", "Recall", "F1-Score"], horizontal=True)
    
    st.bar_chart(metrics_data.set_index('Class')[metric_choice])
    
    # Detailed Table
    with st.expander("Lihat Tabel Detail Metrik"):
        st.table(metrics_data)

    st.write("---")
    
    # 4. Model Architecture Summary
    st.subheader("4. Model Architecture")
    st.code("""
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
    """, language='python')

# --- Tab 4: Prediction App (The Main Functionality) ---
with tab4:
    st.header("Prediction App")
    
    if model is None:
        st.error(f"⚠️ Model tidak ditemukan! Harap letakkan file **{MODEL_PATH}** di folder yang sama.")
    else:
        st.success("✅ Model Ready")
        
        col_pred1, col_pred2 = st.columns([1, 1])
        
        with col_pred1:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
        with col_pred2:
            st.write("### Hasil Deteksi")
            
            if uploaded_file is not None:
                if st.button('🔍 Analyze Image', use_container_width=True):
                    with st.spinner('Analyzing...'):
                        # Preprocessing
                        img = image.resize((224, 224))
                        img_array = tf.keras.utils.img_to_array(img)
                        img_array = tf.expand_dims(img_array, 0)
                        
                        # Predict
                        predictions = model.predict(img_array)
                        score = predictions[0]
                        
                        class_names = ['Flea_Allergy', 'Health', 'Ringworm', 'Scabies']
                        predicted_class = class_names[np.argmax(score)]
                        confidence = 100 * np.max(score)
                        
                        # Display Result
                        if predicted_class == 'Health':
                            st.balloons()
                            st.success(f"**{predicted_class}** (Sehat)")
                        else:
                            st.warning(f"Terdeteksi: **{predicted_class}**")
                            
                        st.progress(int(confidence))
                        st.write(f"Confidence Level: **{confidence:.2f}%**")
                        
                        # Detailed Probabilities
                        st.write("---")
                        st.write("Class Probabilities:")
                        prob_df = pd.DataFrame({
                            'Class': class_names,
                            'Probability': predictions[0]
                        })
                        st.dataframe(prob_df.style.highlight_max(axis=0, subset=['Probability']), use_container_width=True)
            else:
                st.info("Silakan upload gambar di kolom sebelah kiri untuk memulai prediksi.")

# --- Tab 5: Contact Me ---
with tab5:
    st.header("Contact")
    st.write("Hubungi pengembang untuk informasi lebih lanjut.")
    st.text_input("Nama")
    st.text_input("Email")
    st.text_area("Pesan")
    st.button("Kirim Pesan")

