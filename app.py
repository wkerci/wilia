import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights
import torch
from PIL import Image
from torchvision import transforms

# Configuração do modelo
st.title("Classificador de Imagens com ResNet50")
st.write("Envie uma imagem e veja qual classe o modelo identifica.")

# Carregando modelo com pesos atualizados
modelo = resnet50(weights=ResNet50_Weights.DEFAULT)
modelo.eval()

# Transformação da imagem para o modelo
transformacao = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Upload de imagem
arquivo = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if arquivo:
    imagem
