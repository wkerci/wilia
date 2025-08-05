import gradio as gr
from transformers import pipeline
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch

# Chatbot
chatbot = pipeline("text-generation", model="gpt2")

def responder(pergunta):
    resposta = chatbot(pergunta, max_length=100, do_sample=True)[0]["generated_text"]
    return resposta

# Reconhecimento de imagem
modelo = models.resnet50(pretrained=True)
modelo.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def reconhecer(imagem):
    imagem_tensor = transform(imagem).unsqueeze(0)
    with torch.no_grad():
        saida = modelo(imagem_tensor)
        _, classe = saida.max(1)
    return f"Classe prevista: {classe.item()}"

def app(pergunta, imagem):
    texto = responder(pergunta)
    imagem_resp = reconhecer(imagem)
    return texto, imagem_resp

interface = gr.Interface(fn=app,
                         inputs=["text", "image"],
                         outputs=["text", "text"],
                         title="IA Total: Chat + Imagem",
                         description="Faça uma pergunta ou envie uma imagem para ser identificada.")

interface.launch()
