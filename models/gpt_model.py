import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from langchain_openai import ChatOpenAI  # Değişen kısım
from dotenv import load_dotenv
import os

# Çevresel değişkenleri yükle (OpenAI API key için)
load_dotenv(dotenv_path=".gitignore/.env")

# Excel dosyasını yükle
@st.cache_data
def load_data():
    df = pd.read_excel("data/dataset_last_update.xlsx", sheet_name="chatbot_user_inputs_clean")
    return df

df = load_data()

# TF-IDF ve NearestNeighbors modelini eğit
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Input'])
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    return vectorizer, nbrs

vectorizer, nbrs = train_model()

# Kullanıcı girdisinden intent'i tahmin et
def predict_intent(user_input):
    user_vec = vectorizer.transform([user_input])
    distances, indices = nbrs.kneighbors(user_vec)
    return df.iloc[indices[0][0]]['Intent']

# ChatGPT ile intent'e uygun yanıt oluştur
def generate_response_with_openai(intent, user_input):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)  # Değişen kısım
    
    prompt = f"""
    **Görev**: Bir açık artırma uygulamasının chatbotu olduğunu düşün.
    Aşağıdaki kullanıcı sorusuna, verilen intent kategorisine uygun bir yanıt ver.

    **Kurallar**:
    1. Yanıt maksimum 2 cümle olsun.
    2. {intent} intent'i için şu tonu kullan: {"dostane" if intent == "Selamlama" else "profesyonel"}
    3. Kullanıcı soru soruyorsa ("nasıl", "nerede" gibi) direkt cevap ver.
    4. Emoji kullanımı: {"👍, 😊" if intent in ["Selamlama", "Onaylama"] else "Hiç kullanma"}.

    **Kullanıcı Mesajı**: "{user_input}"
    **Intent**: {intent}

    **Örnek Yanıtlar**:
    - Selamlama: "Merhaba! Size nasıl yardımcı olabilirim?"
    - Reddetme: "Bu konuda destek veremiyorum, ancak şu sayfadan ilgili bilgiye ulaşabilirsiniz: https://github.com/GhostCavalry "
    """
    
    response = llm.invoke(prompt)
    return response.content

# Streamlit arayüzü
st.title("🤖 Intent Tabanlı Chatbot (OpenAI Destekli)")

user_input = st.chat_input("Mesajınızı yazın...")

if user_input:
    intent = predict_intent(user_input)
    st.write(f"**Tahmin Edilen Intent:** `{intent}`")
    
    with st.spinner("Yanıt oluşturuluyor..."):
        response = generate_response_with_openai(intent, user_input)  # Değişen kısım
        st.write(f"**Yanıt:** {response}")
    
    # Debug: En yakın 3 örneği göster
    st.divider()
    st.subheader("Model Nasıl Karar Verdi?")
    user_vec = vectorizer.transform([user_input])
    distances, indices = nbrs.kneighbors(user_vec, n_neighbors=3)
    
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        st.write(f"{i+1}. `{df.iloc[idx]['Input']}` (Benzerlik: {1-dist:.2f}, Intent: `{df.iloc[idx]['Intent']}`)")