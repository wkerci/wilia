import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torch

# Configurações iniciais
st.set_page_config(page_title="Classificador com IA", layout="centered")
st.title("🖼️ Classificador de Imagens + Chat IA")

# Carrega o modelo de visão computacional
modelo = resnet50(weights=ResNet50_Weights.DEFAULT)
modelo.eval()

# Transforma imagem
transformacao = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Upload da imagem
arquivo = st.file_uploader("📁 Envie uma imagem para classificar...", type=["jpg", "jpeg", "png"])

if arquivo:
    try:
        imagem = Image.open(arquivo).convert("RGB")
        st.image(imagem, caption="📷 Imagem enviada", use_column_width=True)

        entrada = transformacao(imagem).unsqueeze(0)

        with torch.no_grad():
            saida = modelo(entrada)
            indice = saida.argmax().item()

        rotulos = ResNet50_Weights.DEFAULT.meta["categories"]
        classe = rotulos[indice]

        st.success(f"🧠 Classe identificada: **{classe}**")

    except Exception as e:
        st.error(f"⚠️ Erro ao processar a imagem: {e}")
else:
    st.info("Envie uma imagem para realizar a classificação.")

# Campo de pergunta/chat
st.markdown("---")
st.header("💬 Pergunte algo ao app")

pergunta = st.text_input("Escreva sua pergunta aqui:")

if pergunta:
    st.write("🤖 Resposta simulada do app:")
    # Simulação de resposta (pode integrar com chatbot depois)
    if "o que é resnet" in pergunta.lower():
        st.info("ResNet é uma arquitetura de rede neural profunda usada para classificação de imagens. Ela introduz 'skip connections' para facilitar o treinamento de redes muito profundas.")
    elif "quantas classes" in pergunta.lower():
        st.info("O modelo ResNet50 pré-treinado com ImageNet identifica **1000 classes diferentes**.")
    else:
        st.info("Esse app está focado em reconhecimento de imagens. Para perguntas gerais, integração com chatbot pode ser adicionada futuramente!")
