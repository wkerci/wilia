import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torch
import openai
import os

# 📄 Configuração da página
st.set_page_config(page_title="ResNet50 + ChatGPT", layout="centered")
st.title("🖼️ Classificador com ResNet50 + 💬 Perguntas com IA")

# 🌐 API Key OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")  # ou coloque direto: openai.api_key = "sua-chave-aqui"

# ⚙️ Modelo ResNet50
modelo = resnet50(weights=ResNet50_Weights.DEFAULT)
modelo.eval()

# 🔧 Transformações
transformacao = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 📁 Upload de imagem
arquivo = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if arquivo:
    try:
        imagem = Image.open(arquivo).convert("RGB")
        st.image(imagem, caption="Imagem enviada", use_column_width=True)

        entrada = transformacao(imagem).unsqueeze(0)

        with torch.no_grad():
            saida = modelo(entrada)
            indice = saida.argmax().item()

        rotulos = ResNet50_Weights.DEFAULT.meta["categories"]
        classe = rotulos[indice]

        st.success(f"🧠 Classe identificada: **{classe}**")

    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
else:
    st.info("👆 Envie uma imagem para iniciar a classificação.")

# 💬 Campo de Perguntas
st.markdown("---")
st.header("Pergunte algo ao assistente IA")

pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    with st.spinner("Pensando..."):
        try:
            resposta = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": pergunta}]
            )
            texto_resposta = resposta.choices[0].message.content
            st.info(texto_resposta)

        except Exception as e:
            st.error(f"❌ Erro ao conectar com o chatbot: {e}")
