import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from visualizacoes import plot_perceptron

class SimplePerceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate, self.epochs, self.weights, self.bias, self.errors = learning_rate, epochs, None, None, []
    def fit(self, X, y):
        n_samples, n_features = X.shape; self.weights = np.random.normal(0, 0.01, n_features); self.bias = 0; self.errors = []
        for _ in range(self.epochs):
            epoch_errors = 0
            for i in range(n_samples):
                prediction = self.predict(X[i]); error = y[i] - prediction
                if error != 0: self.weights += self.learning_rate * error * X[i]; self.bias += self.learning_rate * error; epoch_errors += 1
            self.errors.append(epoch_errors);
            if epoch_errors == 0: break
    def predict(self, x): return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

def mostrar():
    st.markdown('<h2 class="section-header">Perceptron - O Primeiro Neurônio Artificial</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""<div class="info-box">O <strong>Perceptron</strong> é o modelo mais simples de neurônio artificial, criado por Frank Rosenblatt em 1957. É um classificador binário linear que pode separar dados linearmente separáveis.</div>""", unsafe_allow_html=True)
        st.markdown("### 📐 Funcionamento Matemático")
        st.markdown("""<div class="formula-box"><strong>Soma Ponderada:</strong><br>z = w₁x₁ + w₂x₂ + ... + wₙxₙ + bias<br><br><strong>Função de Ativação (Degrau):</strong><br>y = 1 se z ≥ 0, caso contrário y = 0</div>""", unsafe_allow_html=True)
    with col2:
        st.subheader("🎮 Controles Interativos"); w1 = st.slider("Peso w₁",-2.,2.,.5,.1); w2 = st.slider("Peso w₂",-2.,2.,-.3,.1); bias = st.slider("Bias",-2.,2.,0.,.1)
        np.random.seed(42); X_example, y_example = make_classification(n_samples=50,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,random_state=1)
    
    st.subheader("📊 Visualização do Perceptron (Manipulação Manual)"); st.markdown("Use os sliders acima para ver como os pesos e o bias afetam a arquitetura e a linha de decisão.")
    st.pyplot(plot_perceptron([w1, w2], bias, X_example, y_example))
    
    st.markdown("---"); st.subheader("🧪 Treinamento do Perceptron"); col1_train, col2_train = st.columns([1, 1])
    with col1_train:
        st.markdown("**Configurações de Treinamento:**"); learning_rate = st.slider("Taxa de Aprendizado",.01,1.,.1,.01,key='lr_p'); epochs = st.slider("Número de Épocas",10,200,50,10,key='ep_p')
        if st.button("🚀 Treinar Perceptron"):
            X_train, y_train = make_classification(n_samples=100,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,random_state=42)
            perceptron = SimplePerceptron(learning_rate=learning_rate,epochs=epochs); perceptron.fit(X_train,y_train)
            st.session_state.perceptron, st.session_state.X_train_p, st.session_state.y_train_p = perceptron, X_train, y_train
    
    with col2_train:
        if 'perceptron' in st.session_state:
            st.markdown("**Resultados do Treinamento:**"); perceptron,X_train,y_train = st.session_state.perceptron,st.session_state.X_train_p,st.session_state.y_train_p
            fig_conv, ax = plt.subplots(figsize=(8,4)); ax.plot(range(1,len(perceptron.errors)+1),perceptron.errors,marker='o'); ax.set_xlabel('Época'); ax.set_ylabel('Número de Erros'); ax.set_title('Convergência do Perceptron'); ax.grid(True,alpha=0.3); st.pyplot(fig_conv)
            st.pyplot(plot_perceptron(perceptron.weights,perceptron.bias,X_train,y_train,title="Resultado Final do Treinamento"))