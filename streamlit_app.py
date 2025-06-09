import streamlit as st
from PIL import Image
from ultralytics import YOLO


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

st.title("YOLOv11 & YOLOv12 Object Detection")


model_option = st.selectbox(
    "Choose a model:",
    ("best-yolov11.pt", "best-YOLOv12.pt")
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        model = load_model(model_option)
        results = model(image)
        result_image = results[0].plot()  # with bounding boxes
        st.image(result_image, caption="Detection Result", use_column_width=True)
