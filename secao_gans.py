# secao_gans.py
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from gans_utils import Generator, LATENT_SIZE

# --- FUNÇÃO PARA CARREGAR O GERADOR ---
@st.cache_resource
def carregar_gerador(caminho_modelo):
    """Carrega o modelo gerador pré-treinado."""
    modelo = Generator()
    # Carrega o estado do modelo, mapeando para CPU caso tenha sido treinado em GPU
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=torch.device('cpu')))
    modelo.eval() # Coloca o modelo em modo de avaliação
    return modelo

def mostrar_imagem_gerada(gerador, noise):
    """Gera e exibe uma imagem a partir de um vetor de ruído."""
    with torch.no_grad():
        imagem_falsa = gerador(noise).detach().cpu()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(imagem_falsa[0, 0], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

def mostrar():
    st.markdown('<h2 class="section-header">🎨 GANs: A Arte de Gerar Imagens</h2>', unsafe_allow_html=True)
    
    try:
        gerador = carregar_gerador('assets/gan/generator_final.pth')
    except FileNotFoundError:
        st.error(
            "**Arquivo do modelo não encontrado!**\n\n"
            "Por favor, execute o [notebook do Google Colab](https://colab.research.google.com/drive/1yZtq7y_g2Q3Qy0G1L85vV8yHkQ6H9p1?usp=sharing) "
            "para treinar o modelo, baixe o `gan_assets.zip` e extraia seu conteúdo para a pasta `assets/gan/` do seu projeto."
        )
        return

    # --- PARTE 1: O CONCEITO ---
    st.markdown("""
    As **Redes Adversárias Generativas (GANs)** são uma das ideias mais fascinantes em Machine Learning. Elas consistem em duas redes neurais que competem em um jogo:
    
    - **🎨 O Gerador (Falsificador):** Sua missão é criar dados falsos (neste caso, imagens de dígitos) que sejam indistinguíveis dos dados reais.
    - **🕵️ O Discriminador (Detetive):** Sua missão é identificar corretamente se uma imagem é real (do dataset MNIST) ou falsa (criada pelo Gerador).
    
    Ambos treinam juntos. O Gerador melhora ao enganar o Discriminador, e o Discriminador melhora ao detectar as fraudes do Gerador. O resultado desse "jogo" é um Gerador extremamente habilidoso em criar dados realistas.
    """)
    
    
    st.markdown("---")
    
    # --- PARTE 2: A JORNADA DO TREINAMENTO ---
    st.subheader("Visualizando o Aprendizado")
    st.markdown("Abaixo, você pode ver a evolução do Gerador ao longo do treinamento. No início, ele produz apenas ruído. Ao final, ele aprendeu a gerar dígitos convincentes.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Evolução das Imagens Geradas")
        st.image("assets/gan/gan_training_progress.gif", caption="Imagens geradas por um mesmo vetor de ruído em diferentes épocas.")
    
    with col2:
        st.markdown("#### O Jogo Adversário (Perdas)")
        st.image("assets/gan/loss_plot.png", caption="Loss do Gerador (G) e Discriminador (D).")
        st.info("💡 **Objetivo:** O Gerador melhora quando sua perda (`loss_g`) diminui. O Discriminador melhora quando sua perda (`loss_d`) é baixa. O equilíbrio ideal acontece quando o Discriminador não consegue mais diferenciar (sua `loss` fica perto de 0.69, ou 50% de acerto).")

    st.markdown("---")

    # --- PARTE 3: PLAYGROUND DO GERADOR ---
    st.subheader("🎮 Playground do Espaço Latente")
    st.markdown("""
    Agora é sua vez de ser o artista! O Gerador transforma um vetor de números aleatórios (o **espaço latente**) em uma imagem. 
    Altere os valores na **barra lateral esquerda** para explorar como pequenas mudanças no ruído de entrada criam diferentes dígitos.
    """)

    # ✅ CORREÇÃO: Adicionando os controles que estavam faltando na barra lateral.
    st.sidebar.subheader("Controles da GAN")
    st.sidebar.markdown("Use estes controles para manipular a imagem gerada em tempo real.")
    
    # Controle para a semente aleatória, que define o "ponto de partida" do ruído
    seed = st.sidebar.slider("Semente Aleatória (Seed)", 1, 100, 42, help="Mude a semente para gerar um dígito completamente novo e, em seguida, ajuste suas características.")
    
    st.sidebar.markdown("Ajuste as primeiras 5 'características' do dígito:")
    
    # Sliders para as 5 primeiras dimensões do vetor latente
    dim_controles = []
    for i in range(5):
        # Usamos st.sidebar para colocar os sliders na barra lateral
        dim_controles.append(st.sidebar.slider(f"Característica {i+1}", -2.0, 2.0, 0.0, 0.1))

    # Gera o vetor de ruído com base nos controles do usuário
    torch.manual_seed(seed) # Garante que o resto do ruído (dimensões 6 a 100) seja consistente
    noise_vector = torch.randn(1, LATENT_SIZE, 1, 1)
    
    # Substitui as primeiras 5 dimensões do ruído pelos valores dos sliders
    for i, val in enumerate(dim_controles):
        noise_vector[0, i, 0, 0] = val

    # Exibe a imagem gerada
    st.markdown("#### Imagem Gerada em Tempo Real")
    mostrar_imagem_gerada(gerador, noise_vector)
    
    with st.expander("O que é o Espaço Latente?"):
        st.markdown("""
        O **espaço latente** é uma representação comprimida e abstrata dos dados que o Gerador aprendeu. Cada dimensão nesse espaço pode controlar uma característica da imagem (como a inclinação de um '7', a espessura de um '1' ou a abertura de um '9').
        
        Ao navegar por este espaço usando os sliders, você está efetivamente pedindo ao Gerador para misturar as características que ele aprendeu e criar novas amostras. É a base da criatividade dos modelos generativos!
        """)