import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Deteksi Objek Multi-Model YOLO",
    page_icon="🤖",
    layout="wide"
)

# --- Judul dan Deskripsi ---
st.title("🚀 Aplikasi Deteksi Objek dengan Multi-Model YOLO")
st.write("Pilih model, unggah gambar, dan biarkan AI mendeteksi objek untuk Anda.")

# --- Pilihan Model ---
# Buat daftar model yang bisa dipilih.
# Kunci (key) adalah nama yang akan ditampilkan di UI.
# Nilai (value) adalah path ke file model.
MODEL_DICT = {
    "YOLOv11": "best-yolov11.pt",
    "YOLOv12": "best-YOLOv12.pt",
}

# Biarkan pengguna memilih model dari selectbox
model_name = st.selectbox(
    "Pilih Model YOLO yang ingin digunakan:",
    list(MODEL_DICT.keys())
)

model_path = MODEL_DICT[model_name]

# --- Memuat Model YOLO dengan Caching ---
# @st.cache_resource digunakan agar model tidak perlu di-load ulang setiap kali 
# ada interaksi user, sehingga lebih cepat dan hemat memori.
@st.cache_resource
def load_yolo_model(path):
    """Memuat model YOLO dari path yang diberikan."""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error memuat model dari '{path}': {e}")
        return None

# Memuat model yang dipilih
model = load_yolo_model(model_path)

if model is None:
    st.stop() # Hentikan eksekusi jika model gagal dimuat

# --- Fungsi untuk melakukan deteksi ---
def detect_objects(image, model_to_use):
    """
    Menerima gambar PIL, melakukan deteksi dengan model yang diberikan, 
    dan mengembalikan gambar dengan bounding box.
    """
    img_array = np.array(image)
    results = model_to_use(img_array)
    annotated_image_bgr = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
    return annotated_image_rgb

# --- Komponen Upload Gambar ---
uploaded_file = st.file_uploader(
    "Pilih sebuah gambar...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.info(f"Model yang digunakan: **{model_name}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("🎯 Hasil Deteksi")
        with st.spinner('Model sedang bekerja...'):
            annotated_image = detect_objects(image, model)
            st.image(annotated_image, use_column_width=True)
else:
    st.info("Silakan unggah file gambar untuk memulai deteksi.")
