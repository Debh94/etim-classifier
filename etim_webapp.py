import streamlit as st
import openai
import os

st.set_page_config(page_title="Assistente AI ETIM con ChatGPT", layout="centered")

st.title("🧠 Assistente AI - Cos'è questo oggetto?")
st.markdown("Scrivi una parola o frase per ottenere **una spiegazione intelligente** dell'oggetto e suggerimenti su dove cercarlo nel classificatore ETIM.")

# Verifica presenza API Key
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ API key OpenAI mancante. Inseriscila in Streamlit Secrets con chiave 'OPENAI_API_KEY'.")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

user_input = st.text_input("🔍 Scrivi qui la tua descrizione, codice o nome commerciale:")

if user_input.strip():
    prompt = f"""
Sei un esperto tecnico che lavora con la classificazione di prodotti del settore edilizia e architettura, in particolare con la classificazione ETIM.

L'utente ti chiede cosa significa un oggetto con questa descrizione: "{user_input.strip()}".

Devi rispondere:
1. Con una **breve definizione tecnica** dell'oggetto
2. Con **esempi di utilizzo**
3. Con un **suggerimento su come cercarlo nel classificatore ETIM**, anche stimando il tipo di famiglia merceologica.

La risposta deve essere semplice, diretta e utile a chi non è esperto.
"""

    with st.spinner("🤖 Sto cercando di interpretare l'oggetto..."):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sei un assistente tecnico esperto di classificazione ETIM."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=500
            )

            answer = response.choices[0].message.content
            st.success("✅ Ecco la spiegazione generata dall'assistente:")
            st.markdown(answer)

        except Exception as e:
            st.error(f"Errore durante la richiesta a OpenAI: {e}")
