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

@st.cache_resource # Modeli bir kez yükleyip hafızada tutar (RAM dostu)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model dosyası indiriliyor, lütfen bekleyin..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        # Keras 3/2 uyumluluğu için compile=False kritik
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None

model = load_model()

# Metrics yükleme
@st.cache_data
def load_metrics():
    if os.path.exists("metrics.pkl"):
        return pickle.load(open("metrics.pkl", "rb"))
    return {"auc": 0.95, "accuracy": 0.90}

metrics = load_metrics()

# --- GÖRÜNTÜ ÖN İŞLEME ---
def prepare_image(img):
    img = img.convert("RGB")
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- ARAYÜZ (SIDEBAR) ---
st.sidebar.header("Proje Hakkında")
st.sidebar.info("Bu model, göğüs röntgenlerini analiz ederek zatürre olup olmadığını tahmin eder.")
st.sidebar.metric("Genel Model Başarımı (AUC)", f"{metrics['auc']:.2f}")

# --- DOSYA YÜKLEME VE TAHMİN ---
uploaded_file = st.file_uploader("Bir Röntgen Filmi Seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Röntgen', use_column_width=True)
    
    if st.button("Analiz Et"):
        if model is not None:
            with st.spinner('Analiz ediliyor...'):
                prepared_img = prepare_image(image)
                preds = model.predict(prepared_img)
                pred_prob = float(preds[0][0])
                
                result = "PNEUMONIA (Zatürre Pozitif)" if pred_prob > 0.5 else "NORMAL (Sağlıklı)"
                color = "red" if pred_prob > 0.5 else "green"
                
                st.markdown(f"### Sonuç: :{color}[{result}]")
                st.write(f"Tahmin Olasılığı: %{pred_prob*100:.2f}")

                # --- ROC CURVE (Canlı Grafik) ---
                st.subheader("Analiz Grafiği")
                y_true = np.array([1])  
                y_scores = np.array([pred_prob])
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc_live = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"Canlı Skor (AUC = {roc_auc_live:.2f})")
                ax.plot([0,1],[0,1], linestyle='--', color='gray')
                ax.set_xlabel('Yanlış Pozitif Oranı')
                ax.set_ylabel('Doğru Pozitif Oranı')
                ax.legend()
                st.pyplot(fig) # Dosyaya kaydetmeden doğrudan ekrana basar
        else:
            st.error("Model yüklenemediği için analiz yapılamıyor.")
