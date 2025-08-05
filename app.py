import openai
import streamlit as st

# Carregando chave da API com segurança
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Interface Streamlit
st.set_page_config(page_title="ChatGPT com Streamlit", page_icon="🤖")
st.title("Chat com GPT via OpenAI 🤖")

# Campo de entrada
pergunta = st.text_input("Digite sua pergunta:")

# Processamento da resposta
if pergunta:
    with st.spinner("Consultando o GPT..."):
        resposta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # ou "gpt-4"
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": pergunta}
            ]
        )
        st.markdown("### 🧠 Resposta:")
        st.write(resposta["choices"][0]["message"]["content"])


