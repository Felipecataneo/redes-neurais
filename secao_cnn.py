import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def mostrar():
    st.markdown('<h2 class="section-header">Redes Neurais Convolucionais (CNNs)</h2>', unsafe_allow_html=True)
    st.markdown("CNNs são uma classe de redes neurais especializada no processamento de dados com uma topologia de grade, como imagens.");
    with st.expander("👁️ A Operação de Convolução",expanded=True):
        st.markdown("A convolução aplica um **filtro (ou kernel)** sobre a imagem, criando um **mapa de características** que destaca padrões como bordas ou texturas.")
        col1,col2 = st.columns([1,2]);
        with col1:
            st.subheader("Filtros Interativos")
            image_data=np.zeros((10,10));
            image_data[2:8,2:8]=10;
            kernels={"Detector de Borda Vertical":np.array([[1,0,-1],[2,0,-2],[1,0,-1]]),"Detector de Borda Horizontal":np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),"Sharpen":np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])};
            kernel_choice=st.selectbox("Escolha um Kernel:",list(kernels.keys()));
            kernel=kernels[kernel_choice]
        with col2:
            convolved_image = convolve2d(image_data,kernel,mode='valid');
            fig,ax = plt.subplots(1,3,figsize=(12,4));
            ax[0].imshow(image_data,cmap='gray');ax[0].set_title('Imagem de Entrada');
            ax[1].imshow(kernel,cmap='gray');ax[1].set_title('Kernel');
            ax[2].imshow(convolved_image,cmap='gray');ax[2].set_title('Mapa de Características');
            for a in ax: a.axis('off');
            st.pyplot(fig)