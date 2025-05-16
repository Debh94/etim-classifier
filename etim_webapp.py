import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_etim_data():
    df = pd.read_excel("Classi_9.xlsx", sheet_name="Foglio2")
    df = df[['Code', 'Description (EN)', 'ETIM IT', 'Translation (ETIM CH)',
             'Traduttore Google', 'Traduzione_DEF', 'Sinonimi']].fillna('')

    # Nuova colonna: unione di tutto il testo utile, compresi i sinonimi
    df['combined_text'] = df.apply(lambda row: ' '.join([
        row['Description (EN)'],
        row['ETIM IT'],
        row['Translation (ETIM CH)'],
        row['Traduttore Google'],
        row['Traduzione_DEF'],
        row['Sinonimi']  # <-- qui usiamo anche i sinonimi!
    ]).lower(), axis=1)

    return df

@st.cache_resource
def setup_classifier(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, tfidf_matrix

def classify_description(description, df, vectorizer, tfidf_matrix):
    input_vec = vectorizer.transform([description.lower()])
    similarity = cosine_similarity(input_vec, tfidf_matrix).flatten()
    idx = similarity.argmax()
    result = df.iloc[idx]
    return result, round(similarity[idx] * 100, 2)

st.set_page_config(page_title="Classificatore ETIM", layout="centered")
st.title("🤖 Classificatore automatico ETIM")
st.markdown("Inserisci una **descrizione tecnica di prodotto** per trovare la classe ETIM più adatta.")

df_etim = load_etim_data()
vectorizer, tfidf_matrix = setup_classifier(df_etim)

user_input = st.text_area("Descrizione del prodotto", height=150)

if st.button("Classifica"):
    if user_input.strip():
        result, score = classify_description(user_input, df_etim, vectorizer, tfidf_matrix)
        st.success(f"✅ Classe ETIM suggerita: **{result['Code']}**")
        st.markdown(f"**Nome (EN):** {result['Description (EN)']}")
        st.markdown(f"**Nome (IT):** {result['ETIM IT']}")
        st.markdown(f"**Sinonimi/Traduzioni:**")
        st.markdown(f"- ETIM CH: {result['Translation (ETIM CH)']}")
        st.markdown(f"- Google Translate: {result['Traduttore Google']}")
        st.markdown(f"- Traduzione DEF: {result['Traduzione_DEF']}")
        st.markdown(f"**Confidenza AI:** {score}%")
    else:
        st.warning("Inserisci una descrizione valida per procedere.")
