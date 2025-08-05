import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torch

# Título do aplicativo
st.set_page_config(page_title="Classificador ResNet50", layout="centered")
st.title("🖼️ Classificador de Imagens com ResNet50")
st.write("Envie uma imagem e veja qual classe o modelo identifica usando a arquitetura ResNet50 do PyTorch.")

# Carrega o modelo com pesos atualizados recomendados
modelo = resnet50(weights=ResNet50_Weights.DEFAULT)
modelo.eval()

# Transformações para adequar a imagem ao modelo
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
arquivo = st.file_uploader("📁 Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if arquivo:
    try:
        imagem = Image.open(arquivo).convert("RGB")
        st.image(imagem, caption="📷 Imagem enviada", use_column_width=True)

        entrada = transformacao(imagem).unsqueeze(0)

        with torch.no_grad():
            saida = modelo(entrada)
            indice = saida.argmax().item()

        # Rótulo da classe
        rotulos = ResNet50_Weights.DEFAULT.meta["categories"]
        classe = rotulos[indice]

        st.success(f"🧠 Classe identificada: **{classe}**")

    except Exception as e:
        st.error(f"⚠️ Erro ao processar a imagem: {e}")

else:
    st.info("Por favor, envie uma imagem para que o modelo possa fazer a classificação.")
