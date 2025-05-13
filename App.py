#streamlit run "D:\Unsri\Semester 6\Pemrosesan Bahasa Alami\App\App.py"

import streamlit as st
import pandas as pd
import nltk
import spacy
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from PIL import Image

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocessing functions (same as training)
def preserve_named_entities(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG"]:
            text = text.replace(ent.text, ent.text.replace(" ", "_"))
    return text

def preprocess_text(text):
    text = preserve_named_entities(text)
    tokens = word_tokenize(text)
    extra_stopwords = {'said', 'also', 'like', 'likely', 'would', 'could'}
    stop_words = set(stopwords.words('english')).union(extra_stopwords)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if not word.isdigit()]
    return ' '.join(tokens)

cluster_labels = {
    0: "Politik",
    1: "Gadget",
    2: "Teknologi Umum & Entertaiment"
}

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\user\Downloads\App\App\preprocessed_data.csv", header=None, names=["text"], encoding="utf-8")
    df["text"] = df["text"].apply(preprocess_text)
    return df

# Streamlit UI

st.title("Clustering Kategori Berita")

image_path = (r"C:\Users\user\Downloads\App\App\cover.jpg")
image = Image.open(image_path)
st.image(image, width=600)

st.write("Masukkan Isi Berita")

user_input = st.text_area("Masukkan teks di sini:")

if st.button("Prediksi Klaster"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        # Load & process dataset
        df = load_data()

        # TF-IDF dan reduksi dimensi
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df["text"])

        svd = TruncatedSVD(n_components=50, random_state=42)
        reduced_matrix = svd.fit_transform(tfidf_matrix)

        # KMeans dengan cluster optimal, sesuai hasilnya, optimalnya ialah 3
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(reduced_matrix)

        # Preprocess input user
        preprocessed = preprocess_text(user_input)
        tfidf_user = vectorizer.transform([preprocessed])
        reduced_user = svd.transform(tfidf_user)
        cluster = kmeans.predict(reduced_user)[0]
        kategori = cluster_labels.get(cluster, "Tidak diketahui")

        st.success(f"Berita ini masuk ke **Cluster {cluster}** - Kategori: **{kategori}**")