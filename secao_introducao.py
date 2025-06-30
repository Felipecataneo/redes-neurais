import streamlit as st

def mostrar():
    st.markdown('<h2 class="section-header">O que são Redes Neurais?</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""<div class="info-box">As <strong>Redes Neurais Artificiais (RNAs)</strong> são modelos computacionais inspirados pela estrutura e funcionamento do cérebro humano. Elas constituem o núcleo de muitos dos avanços em Inteligência Artificial e são capazes de aprender padrões complexos a partir de dados através de um processo de treinamento.<h4>Componentes Fundamentais:</h4><ul><li><b>Neurônios (ou Nós):</b> As unidades computacionais básicas que recebem entradas, processam-nas e geram uma saída.</li><li><b>Conexões e Pesos:</b> Cada conexão entre neurônios possui um peso associado, que modula a força do sinal. O aprendizado ocorre pelo ajuste desses pesos.</li><li><b>Bias:</b> Um parâmetro extra, similar ao intercepto em uma regressão linear, que permite deslocar a função de ativação.</li><li><b>Função de Ativação:</b> Determina a saída do neurônio, introduzindo não-linearidades que permitem à rede aprender padrões complexos.</li></ul></div>""", unsafe_allow_html=True)
    with col2:
        st.image("assets/images.png", caption="Modelo de um neurônio artificial.", use_container_width=True)