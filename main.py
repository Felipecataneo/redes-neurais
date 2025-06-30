# main.py
import streamlit as st

# Importa a função 'mostrar' de cada módulo de seção com um nome único
from secao_introducao import mostrar as mostrar_introducao
from secao_perceptron import mostrar as mostrar_perceptron
from secao_redes_neurais import mostrar as mostrar_redes_neurais
from secao_backpropagation import mostrar as mostrar_backpropagation
from secao_mlp_acao import mostrar as mostrar_mlp_acao
from secao_cnn import mostrar as mostrar_cnn
from secao_rnn import mostrar as mostrar_rnn
from secao_transformers import mostrar as mostrar_transformers
from secao_playground import mostrar as mostrar_playground
from secao_gans import mostrar as mostrar_gans

def main():
    """Função principal que executa o aplicativo Streamlit."""
    
    # --- CONFIGURAÇÃO DA PÁGINA E CSS ---
    st.set_page_config(page_title="Guia Didático de Redes Neurais", page_icon="🧠", layout="wide")

    st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 2.2rem; color: #ff7f0e; border-bottom: 2px solid #ff7f0e; padding-bottom: .5rem; margin: 2rem 0 1rem; }
    .info-box { background-color: #f0f8ff; padding: 1rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 1rem 0; }
    .formula-box { background-color: #fff8dc; padding: 1rem; border-radius: 10px; border: 2px solid #ffa500; margin: 1rem 0; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🧠 Guia Didático Interativo de Redes Neurais</h1>', unsafe_allow_html=True)

    # --- NAVEGAÇÃO (SIDEBAR) ---
    st.sidebar.title("📚 Navegação")

    # Mapeia os nomes das seções para as funções que as renderizam
    PAGINAS = {
        "🔍 Introdução": mostrar_introducao,
        "🔧 Perceptron": mostrar_perceptron,
        "🌐 Redes Neurais": mostrar_redes_neurais,
        "🔄 Backpropagation": mostrar_backpropagation,
        "🧠 MLP em Ação": mostrar_mlp_acao,
        "🖼️ CNNs": mostrar_cnn,
        "📜 RNNs": mostrar_rnn,
        "🎨 GANs": mostrar_gans,
        "🤖 Transformers": mostrar_transformers,
        "🎮 Playground Interativo": mostrar_playground,
    }

    secao = st.sidebar.radio("Escolha uma seção:", list(PAGINAS.keys()))

    # --- RENDERIZA A PÁGINA SELECIONADA ---
    # Chama a função correspondente à seção escolhida no rádio button
    pagina_selecionada = PAGINAS[secao]
    pagina_selecionada()


    # --- RODAPÉ ---
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'><strong>Guia Didático de Redes Neurais</strong> | Desenvolvido para o aprendizado aprofundado de conceitos de Machine Learning.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()