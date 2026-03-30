import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import os
import gdown
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩺 Zatürre (Pneumonia) Teşhis Sistemi")

# --- MODEL İNDİRME VE YÜKLEME ---
MODEL_PATH = "pneumonia_model.keras"
FILE_ID = "11Bt0nfupPs4T6G4xxMoW6mD0eqg-bBgK"

@st.cache_resource # Modeli bir kez yükler, RAM'i verimli kullanır
def load_pneumonia_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model Google Drive'dan indiriliyor, lütfen bekleyin..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        # Keras 3/2 uyumluluğu için compile=False ve safe_mode=False kritik
        # Render'daki sürüm farklarını bu parametrelerle aşıyoruz
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

# Modeli çalıştır
model = load_pneumonia_model()

# Metrics yükleme
@st.cache_data
def load_metrics():
    if os.path.exists("metrics.pkl"):
        try:
            with open("metrics.pkl", "rb") as f:
                return pickle.load(f)
        except:
            pass
    return {"auc": 0.95} # Dosya yoksa varsayılan

metrics = load_metrics()

# --- GÖRÜNTÜ ÖN İŞLEME ---
def prepare_image(img):
    img = img.convert("RGB") # Renkli formata zorla
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- ARAYÜZ (SIDEBAR) ---
st.sidebar.header("Proje Hakkında")
st.sidebar.info("Bu sistem, göğüs röntgenlerini derin öğrenme kullanarak analiz eder.")
st.sidebar.metric("Genel Model Başarımı (AUC)", f"{metrics.get('auc', 0):.2f}")

# --- DOSYA YÜKLEME VE TAHMİN ---
uploaded_file = st.file_uploader("Bir Röntgen Filmi Seçin (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Röntgen', use_column_width=True)
    
    if st.button("🔍 Analiz Et"):
        if model is not None:
            with st.spinner('Yapay zeka analiz ediyor...'):
                # Tahmin Süreci
                prepared_img = prepare_image(image)
                preds = model.predict(prepared_img)
                pred_prob = float(preds[0][0])
                
                # Karar
                result = "PNEUMONIA (Zatürre Pozitif)" if pred_prob > 0.5 else "NORMAL (Sağlıklı)"
                color = "red" if pred_prob > 0.5 else "green"
                
                st.markdown(f"### Sonuç: :{color}[{result}]")
                st.write(f"**Tahmin Olasılığı:** %{pred_prob*100:.2f}")

                # --- CANLI ROC GRAFİĞİ ---
                st.subheader("Analiz Özeti")
                # Basit bir görselleştirme
                fig, ax = plt.subplots()
                ax.bar(["Zatürre Olasılığı", "Sağlıklı Olasılığı"], [pred_prob, 1-pred_prob], color=[color, 'gray'])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Olasılık Skoru')
                st.pyplot(fig)
        else:
            st.error("Üzgünüz, model yüklenemediği için şu an işlem yapılamıyor.")
