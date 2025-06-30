import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from visualizacoes import plot_neural_network

def mostrar():
    st.markdown('<h2 class="section-header">Redes Neurais Multicamadas (MLP)</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1]);
    with col1:
        st.markdown("""<div class="info-box">As <strong>Redes Neurais Multicamadas</strong> (MLPs - Multi-Layer Perceptrons) superam as limitações do perceptron simples, sendo capazes de resolver problemas não-linearmente separáveis através de camadas ocultas e funções de ativação não-lineares.</div>""", unsafe_allow_html=True)
        st.markdown("### 🏗️ Arquitetura\n- **Camada de Entrada**: Recebe os dados brutos.\n- **Camadas Ocultas**: Processam e extraem características dos dados.\n- **Camada de Saída**: Produz o resultado final.")
    with col2:
        st.subheader("🎮 Configurar Arquitetura"); input_neurons = st.number_input("Neurônios de Entrada",2,10,3,key='nn_in'); hidden_neurons = st.number_input("Neurônios Ocultos",2,20,5,key='nn_hid'); output_neurons = st.number_input("Neurônios de Saída",1,5,1,key='nn_out'); layers = [input_neurons, hidden_neurons, output_neurons]
    
    st.subheader("📊 Visualização da Arquitetura"); st.markdown("Use os controles acima para montar a arquitetura da sua rede."); st.pyplot(plot_neural_network(layers))
    
    st.markdown("---"); st.subheader("📈 Funções de Ativação Comuns"); st.markdown("Essas funções introduzem não-linearidade, permitindo que a rede aprenda padrões complexos."); x = np.linspace(-5, 5, 100); sigmoid = 1/(1+np.exp(-x)); relu = np.maximum(0,x); tanh = np.tanh(x)
    fig_activation, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,8)); fig_activation.suptitle("Visualização de Funções de Ativação",fontsize=16)
    ax1.plot(x,sigmoid,'b-',linewidth=2); ax1.set_title('Sigmoid'); ax1.grid(True,alpha=0.3); ax1.set_ylim(-.1,1.1)
    ax2.plot(x,relu,'r-',linewidth=2); ax2.set_title('ReLU'); ax2.grid(True,alpha=0.3)
    ax3.plot(x,tanh,'g-',linewidth=2); ax3.set_title('Tanh'); ax3.grid(True,alpha=0.3); ax3.set_ylim(-1.1,1.1)
    ax4.plot(x,sigmoid,'b-',label='Sigmoid'); ax4.plot(x,relu,'r-',label='ReLU'); ax4.plot(x,tanh,'g-',label='Tanh'); ax4.set_title('Comparação'); ax4.legend(); ax4.grid(True,alpha=0.3)
    plt.tight_layout(rect=[0,0.03,1,0.95]); st.pyplot(fig_activation)