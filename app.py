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

        rotulos = ResNet50_Weights.DEFAULT.meta["categories"]
        classe = rotulos[indice]

        st.success(f"🧠 Classe identificada: **{classe}**")

    except Exception as e:
        st.error(f"⚠️ Erro ao processar a imagem: {e}")
else:
    st.info("Envie uma imagem para realizar a classificação.")

# Campo de perguntas
st.markdown("---")
st.header("💬 Campo de Perguntas")

pergunta = st.text_input("Escreva sua pergunta:")

if pergunta:
    st.write("🤖 Resposta do app:")
    
    # Simulação de resposta baseada no texto
    pergunta_lower = pergunta.lower()

    if "o que é resnet" in pergunta_lower:
        st.info("ResNet é uma rede neural profunda que usa conexões residuais para facilitar o treinamento de redes com muitas camadas.")
    elif "quantas classes" in pergunta_lower:
        st.info("O modelo ResNet50 pré-treinado com ImageNet reconhece **1000 classes diferentes**.")
    elif "como funciona" in pergunta_lower:
        st.info("O modelo transforma a imagem em tensores, passa por camadas convolucionais e gera probabilidades para cada classe. A de maior valor é escolhida.")
    else:
        st.info("Esse app está focado em reconhecimento de imagens. Para perguntas gerais, podemos integrar um chatbot futuramente!")
