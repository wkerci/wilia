import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torch

# Configuração da página
st.set_page_config(page_title="ResNet50 + Chat IA", layout="centered")
st.title("🖼️ Classificador de Imagens com ResNet50")
st.write("Envie uma imagem para descobrir a classe identificada e envie perguntas ao app!")

# Carrega o modelo com pesos recomendados
modelo = resnet50(weights=ResNet50_Weights.DEFAULT)
modelo.eval()

# Transformações para tratar a imagem
transformacao = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
