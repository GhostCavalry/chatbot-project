import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Veri Yükleme
df = pd.read_excel("data/dataset_last_update.xlsx")
X = df['Input']
y = df['Intent']

# 2. Metin Ön İşleme
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

X = X.apply(preprocess_text)

# 3. Train-Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TF-IDF Vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 5. Model Performansını Hesaplama
def evaluate_model(model, model_name):
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)  # Confusion matrix hesaplama
    return {"Model": model_name, "Precision": precision, "Recall": recall, "F1 Score": f1, "Confusion Matrix": cm}

# 6. Modelleri Karşılaştırma
results = []

# GPT Modeli (örnek olarak RandomForest kullanılıyor)
gpt_model = RandomForestClassifier(random_state=42)
results.append(evaluate_model(gpt_model, "GPT"))

# Gemini Modeli (örnek olarak başka bir RandomForest kullanılıyor)
gemini_model = RandomForestClassifier(random_state=42)
results.append(evaluate_model(gemini_model, "Gemini"))

# 7. Sonuçları Tablo Haline Getirme
results_df = pd.DataFrame(results)

# Confusion Matrix'leri ayrı ayrı yazdırma
for result in results:
    print(f"Model: {result['Model']}")
    print("Confusion Matrix:")
    print(result["Confusion Matrix"])
    print()

# Performans metriklerini yazdırma
print(results_df.drop(columns=["Confusion Matrix"]))  # Confusion Matrix sütununu tablodan çıkararak yazdır