import streamlit as st
st.set_page_config(page_title="Pet Segmentation", layout="centered")

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import os
import io

from model.unet import UNET
from config import DEVICE

# Đường dẫn đến model đã huấn luyện
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")

# Load mô hình

@st.cache_resource
def load_model():
    model = UNET().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# Tiền xử lý ảnh đầu vào
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),   # Resize cho phù hợp với input size khi train
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# Dự đoán phân vùng
def predict(image):
    with torch.no_grad():
        start = time.time()
        input_tensor = preprocess(image)
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        end = time.time()
    return pred, end - start

# Chuyển mask thành ảnh Grayscale (đen-xám-trắng)
def mask_to_grayscale(mask):
    gray_values = np.array([0, 127, 255], dtype=np.uint8)  # class 0 → black, 1 → gray, 2 → white
    gray_mask = gray_values[mask]
    return Image.fromarray(gray_mask)

# Giao diện Streamlit
st.title("🐶🐱 Pet Image Segmentation Web")

# Khởi tạo session_state nếu chưa có
if "result_image" not in st.session_state:
    st.session_state["result_image"] = None
    st.session_state["result_bytes"] = None
    st.session_state["proc_time"] = 0.0

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Process Image"):
        st.write("⏳ Processing...")
        seg_mask, proc_time = predict(image)
        result = mask_to_grayscale(seg_mask)

        # Lưu kết quả vào session_state
        st.session_state["result_image"] = result
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.session_state["result_bytes"] = buf.getvalue()
        st.session_state["proc_time"] = proc_time

# Hiển thị kết quả nếu có
if st.session_state["result_image"]:
    st.image(
        st.session_state["result_image"],
        caption=f"Segmented Output (Time: {st.session_state['proc_time']:.2f}s)",
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Segmentation Result",
        data=st.session_state["result_bytes"],
        file_name="segmented_result.png",
        mime="image/png"
    )
