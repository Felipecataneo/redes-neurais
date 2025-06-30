import streamlit as st

def mostrar():
    st.markdown('<h2 class="section-header">Redes Neurais Recorrentes (RNNs)</h2>', unsafe_allow_html=True)
    st.markdown("RNNs são projetadas para trabalhar com dados sequenciais. Sua característica definidora é a **conexão recorrente**, que cria uma **memória** para reter informações sobre o passado.")
    st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png",caption="Uma RNN 'desdobrada' no tempo. Fonte: Chris Olah's Blog.")
    with st.expander("🧠 Solução para Memória de Longo Prazo: LSTM e GRU",expanded=True):
        st.markdown("**Long Short-Term Memory (LSTM)** e **Gated Recurrent Unit (GRU)** usam 'portões' (gates) para regular o fluxo de informação, permitindo que a rede aprenda a reter ou descartar informações de forma seletiva.")
        st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png",caption="A estrutura de uma célula LSTM com seus portões. Fonte: Chris Olah's Blog.")