import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification, make_circles
from sklearn.preprocessing import StandardScaler

# Configuração da página
st.set_page_config(
    page_title="Redes Neurais - Aprenda de Forma Didática",
    page_icon="🧠",
    layout="wide"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .formula-box {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ffa500;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🧠 Redes Neurais - Guia Didático Interativo</h1>', unsafe_allow_html=True)

# Sidebar para navegação
st.sidebar.title("📚 Navegação")
secao = st.sidebar.radio(
    "Escolha uma seção:",
    ["🔍 Introdução", "🔧 Perceptron", "🌐 Redes Neurais", "🔄 Backpropagation", "🎮 Playground Interativo"]
)

# Função para criar visualização do perceptron
def plot_perceptron(weights, bias, X, y, title="Perceptron"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Arquitetura do Perceptron
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.set_title("Arquitetura do Perceptron", fontsize=14, fontweight='bold')
    
    # Entradas
    for i in range(2):
        circle = Circle((1, 2 + i * 2), 0.3, color='lightblue', ec='black')
        ax1.add_patch(circle)
        ax1.text(1, 2 + i * 2, f'x{i+1}', ha='center', va='center', fontweight='bold')
        
        # Setas com pesos
        ax1.arrow(1.3, 2 + i * 2, 3.4, 1 - i * 2, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax1.text(3, 2.5 + i * 0.5, f'w{i+1}={weights[i]:.2f}', fontsize=10, color='red')
    
    # Neurônio
    circle = Circle((5.5, 3), 0.5, color='orange', ec='black')
    ax1.add_patch(circle)
    ax1.text(5.5, 3, 'Σ', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Bias
    ax1.text(5.5, 2.2, f'bias={bias:.2f}', ha='center', fontsize=10, color='blue')
    
    # Saída
    ax1.arrow(6, 3, 2, 0, head_width=0.1, head_length=0.1, fc='green', ec='green')
    circle = Circle((8.5, 3), 0.3, color='lightgreen', ec='black')
    ax1.add_patch(circle)
    ax1.text(8.5, 3, 'y', ha='center', va='center', fontweight='bold')
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Gráfico 2: Dados e linha de decisão
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', label='Classe 0', s=50)
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', label='Classe 1', s=50)
    
    # Linha de decisão
    if weights[1] != 0:
        x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        y_line = -(weights[0] * x_line + bias) / weights[1]
        ax2.plot(x_line, y_line, 'k--', linewidth=2, label='Linha de Decisão')
    
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Classificação do Perceptron')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return fig

# Função para visualizar rede neural
def plot_neural_network(layers):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layer_positions = np.linspace(1, 10, len(layers))
    max_neurons = max(layers)
    
    colors = ['lightblue', 'orange', 'lightgreen']
    
    for i, (pos, neurons) in enumerate(zip(layer_positions, layers)):
        y_positions = np.linspace(1, max_neurons, neurons)
        color = colors[i % len(colors)]
        
        for j, y_pos in enumerate(y_positions):
            circle = Circle((pos, y_pos), 0.2, color=color, ec='black')
            ax.add_patch(circle)
            
            # Conectar com próxima layer
            if i < len(layers) - 1:
                next_y_positions = np.linspace(1, max_neurons, layers[i + 1])
                for next_y in next_y_positions:
                    ax.plot([pos + 0.2, layer_positions[i + 1] - 0.2], 
                           [y_pos, next_y], 'k-', alpha=0.3, linewidth=0.5)
    
    # Labels
    labels = ['Entrada', 'Oculta', 'Saída']
    for i, pos in enumerate(layer_positions):
        if i < len(labels):
            ax.text(pos, max_neurons + 0.5, labels[i], ha='center', fontweight='bold')
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0, max_neurons + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Arquitetura da Rede Neural', fontsize=16, fontweight='bold')
    
    return fig

# Implementação simples do perceptron
class SimplePerceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        self.errors = []
        
        for epoch in range(self.epochs):
            epoch_errors = 0
            for i in range(n_samples):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                    epoch_errors += 1
            
            self.errors.append(epoch_errors)
            
            if epoch_errors == 0:
                break
    
    def predict(self, x):
        activation = np.dot(x, self.weights) + self.bias
        return 1 if activation >= 0 else 0

# Seção: Introdução
if secao == "🔍 Introdução":
    st.markdown('<h2 class="section-header">O que são Redes Neurais?</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        As <strong>Redes Neurais Artificiais</strong> são modelos computacionais inspirados no funcionamento do cérebro humano. 
        Elas são capazes de aprender padrões complexos nos dados através de um processo de treinamento.
        
        <h4>Componentes Principais:</h4>
        • <strong>Neurônios (Nós)</strong>: Unidades de processamento básicas<br>
        • <strong>Conexões (Pesos)</strong>: Determinam a força da ligação entre neurônios<br>
        • <strong>Função de Ativação</strong>: Define a saída do neurônio<br>
        • <strong>Camadas</strong>: Grupos de neurônios organizados hierarquicamente
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://via.placeholder.com/300x200/1f77b4/white?text=Neurônio+Biológico", 
                caption="Inspiração: Neurônio Biológico")
    
    st.markdown("""
    ### 🎯 Por que usar Redes Neurais?
    
    - **Reconhecimento de Padrões**: Excelentes para identificar padrões complexos
    - **Aprendizado Adaptativo**: Melhoram com mais dados
    - **Versatilidade**: Aplicáveis a diversos problemas (classificação, regressão, etc.)
    - **Capacidade de Generalização**: Funcionam bem com dados não vistos anteriormente
    """)

# Seção: Perceptron
elif secao == "🔧 Perceptron":
    st.markdown('<h2 class="section-header">Perceptron - O Primeiro Neurônio Artificial</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        O <strong>Perceptron</strong> é o modelo mais simples de neurônio artificial, criado por Frank Rosenblatt em 1957.
        É um classificador binário linear que pode separar dados linearmente separáveis.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 📐 Funcionamento Matemático
        """)
        
        st.markdown("""
        <div class="formula-box">
        <strong>Soma Ponderada:</strong><br>
        z = w₁x₁ + w₂x₂ + ... + wₙxₙ + bias<br><br>
        <strong>Função de Ativação (Degrau):</strong><br>
        y = 1 se z ≥ 0, caso contrário y = 0
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Controles interativos
        st.subheader("🎮 Controles Interativos")
        w1 = st.slider("Peso w₁", -2.0, 2.0, 0.5, 0.1)
        w2 = st.slider("Peso w₂", -2.0, 2.0, -0.3, 0.1)
        bias = st.slider("Bias", -2.0, 2.0, 0.0, 0.1)
        
        # Gerar dados de exemplo
        np.random.seed(42)
        X_example = np.random.randn(20, 2)
        y_example = (X_example[:, 0] + X_example[:, 1] > 0).astype(int)
    
    # Visualização do perceptron
    st.subheader("📊 Visualização do Perceptron")
    fig = plot_perceptron([w1, w2], bias, X_example, y_example)
    st.pyplot(fig)
    
    # Demonstração prática
    st.subheader("🧪 Treinamento do Perceptron")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Configurações de Treinamento:**")
        learning_rate = st.slider("Taxa de Aprendizado", 0.01, 1.0, 0.1, 0.01)
        epochs = st.slider("Número de Épocas", 10, 200, 50, 10)
        
        if st.button("🚀 Treinar Perceptron"):
            # Criar dados linearmente separáveis
            X_train, y_train = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                                 n_informative=2, n_clusters_per_class=1, 
                                                 random_state=42)
            
            # Treinar perceptron
            perceptron = SimplePerceptron(learning_rate=learning_rate, epochs=epochs)
            perceptron.fit(X_train, y_train)
            
            st.session_state.perceptron_trained = True
            st.session_state.perceptron = perceptron
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
    
    with col2:
        if hasattr(st.session_state, 'perceptron_trained') and st.session_state.perceptron_trained:
            # Mostrar resultado do treinamento
            perceptron = st.session_state.perceptron
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            
            # Gráfico de convergência
            fig_conv, ax = plt.subplots(figsize=(8, 4))
            ax.plot(perceptron.errors)
            ax.set_xlabel('Época')
            ax.set_ylabel('Número de Erros')
            ax.set_title('Convergência do Perceptron')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_conv)
            
            # Resultado final
            fig_result = plot_perceptron(perceptron.weights, perceptron.bias, X_train, y_train)
            st.pyplot(fig_result)

# Seção: Redes Neurais
elif secao == "🌐 Redes Neurais":
    st.markdown('<h2 class="section-header">Redes Neurais Multicamadas</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        As <strong>Redes Neurais Multicamadas</strong> (MLPs - Multi-Layer Perceptrons) superam as limitações do perceptron simples,
        sendo capazes de resolver problemas não-linearmente separáveis através de camadas ocultas.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🏗️ Arquitetura
        
        **Camada de Entrada**: Recebe os dados de entrada
        **Camadas Ocultas**: Processam e transformam as informações
        **Camada de Saída**: Produz o resultado final
        
        ### 🔄 Funções de Ativação Comuns
        
        - **Sigmoid**: σ(x) = 1/(1 + e^(-x))
        - **ReLU**: f(x) = max(0, x)
        - **Tanh**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
        """)
    
    with col2:
        st.subheader("🎮 Configurar Arquitetura")
        
        # Controles para configurar a rede
        input_neurons = st.number_input("Neurônios de Entrada", 2, 10, 3)
        hidden_neurons = st.number_input("Neurônios Ocultos", 2, 20, 5)
        output_neurons = st.number_input("Neurônios de Saída", 1, 5, 1)
        
        layers = [input_neurons, hidden_neurons, output_neurons]
    
    # Visualização da arquitetura
    st.subheader("📊 Visualização da Arquitetura")
    fig_nn = plot_neural_network(layers)
    st.pyplot(fig_nn)
    
    # Comparação de funções de ativação
    st.subheader("📈 Funções de Ativação")
    
    x = np.linspace(-5, 5, 100)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    
    # ReLU
    relu = np.maximum(0, x)
    
    # Tanh
    tanh = np.tanh(x)
    
    fig_activation, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Sigmoid
    ax1.plot(x, sigmoid, 'b-', linewidth=2)
    ax1.set_title('Sigmoid')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # ReLU
    ax2.plot(x, relu, 'r-', linewidth=2)
    ax2.set_title('ReLU')
    ax2.grid(True, alpha=0.3)
    
    # Tanh
    ax3.plot(x, tanh, 'g-', linewidth=2)
    ax3.set_title('Tanh')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)
    
    # Comparação
    ax4.plot(x, sigmoid, 'b-', label='Sigmoid', linewidth=2)
    ax4.plot(x, relu, 'r-', label='ReLU', linewidth=2)
    ax4.plot(x, tanh, 'g-', label='Tanh', linewidth=2)
    ax4.set_title('Comparação')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_activation)

# Seção: Backpropagation
elif secao == "🔄 Backpropagation":
    st.markdown('<h2 class="section-header">Backpropagation - O Algoritmo de Aprendizado</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        O <strong>Backpropagation</strong> é o algoritmo fundamental para treinar redes neurais multicamadas.
        Ele calcula o gradiente da função de erro em relação aos pesos, permitindo o ajuste eficiente dos parâmetros.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔄 Processo do Backpropagation
        
        1. **Forward Pass**: Cálculo da saída da rede
        2. **Cálculo do Erro**: Comparação com o valor esperado
        3. **Backward Pass**: Propagação do erro para trás
        4. **Atualização dos Pesos**: Ajuste baseado no gradiente
        """)
    
    with col2:
        st.markdown("""
        <div class="formula-box">
        <strong>Regra da Cadeia:</strong><br>
        ∂E/∂w = ∂E/∂y × ∂y/∂z × ∂z/∂w<br><br>
        <strong>Atualização do Peso:</strong><br>
        w_novo = w_antigo - η × ∂E/∂w<br><br>
        <em>onde η é a taxa de aprendizado</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualização do processo
    st.subheader("📊 Visualização do Processo de Treinamento")
    
    # Simulação de treinamento
    epochs = 50
    initial_error = 100
    errors = []
    
    for i in range(epochs):
        # Simulação de redução do erro
        error = initial_error * np.exp(-i * 0.1) + np.random.normal(0, 2)
        errors.append(max(0, error))
    
    fig_training, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de erro
    ax1.plot(errors, 'b-', linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Erro')
    ax1.set_title('Redução do Erro Durante o Treinamento')
    ax1.grid(True, alpha=0.3)
    
    # Visualização do gradiente
    x = np.linspace(-3, 3, 100)
    loss = x**2  # Função de perda quadrática
    gradient = 2*x  # Derivada
    
    ax2.plot(x, loss, 'r-', linewidth=2, label='Função de Perda')
    
    # Ponto atual
    current_x = 2
    ax2.plot(current_x, current_x**2, 'ro', markersize=10, label='Ponto Atual')
    
    # Tangente (gradiente)
    tangent_y = gradient[abs(x - current_x).argmin()] * (x - current_x) + current_x**2
    ax2.plot(x, tangent_y, 'g--', linewidth=2, label='Gradiente')
    
    ax2.set_xlabel('Parâmetro (peso)')
    ax2.set_ylabel('Erro')
    ax2.set_title('Descida do Gradiente')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_training)
    
    # Algoritmo passo a passo
    st.subheader("🔍 Algoritmo Passo a Passo")
    
    with st.expander("Ver Pseudocódigo do Backpropagation"):
        st.code("""
        Para cada época de treinamento:
            Para cada exemplo de treinamento (x, y):
                # Forward Pass
                1. Calcular saída de cada camada:
                   a₁ = f(W₁ × x + b₁)
                   a₂ = f(W₂ × a₁ + b₂)
                   ...
                   saída = f(Wₙ × aₙ₋₁ + bₙ)
                
                # Cálculo do Erro
                2. erro = saída - y
                
                # Backward Pass
                3. Para cada camada (da última para a primeira):
                   δ = erro × f'(z)  # delta da camada
                   ∂E/∂W = δ × a_anterior
                   ∂E/∂b = δ
                
                # Atualização dos Pesos
                4. W = W - η × ∂E/∂W
                   b = b - η × ∂E/∂b
        """, language='python')

# Seção: Playground Interativo
elif secao == "🎮 Playground Interativo":
    st.markdown('<h2 class="section-header">Playground Interativo</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Experimente diferentes configurações e veja como elas afetam o desempenho da rede neural!
    </div>
    """, unsafe_allow_html=True)
    
    # Controles do playground
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("🎯 Dados")
        dataset_type = st.selectbox(
            "Tipo de Dataset",
            ["Classificação Linear", "Círculos Concêntricos", "Dataset Customizado"]
        )
        n_samples = st.slider("Número de Amostras", 50, 500, 200)
    
    with col2:
        st.subheader("🏗️ Arquitetura")
        hidden_layers = st.slider("Camadas Ocultas", 1, 3, 1)
        neurons_per_layer = st.slider("Neurônios por Camada", 2, 20, 5)
        activation = st.selectbox("Função de Ativação", ["ReLU", "Sigmoid", "Tanh"])
    
    with col3:
        st.subheader("⚙️ Treinamento")
        learning_rate_pg = st.slider("Taxa de Aprendizado", 0.001, 0.5, 0.01, 0.001)
        epochs_pg = st.slider("Épocas", 10, 500, 100)
    
    # Gerar dados baseado na seleção
    if dataset_type == "Classificação Linear":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
    elif dataset_type == "Círculos Concêntricos":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    else:
        # Dataset customizado (XOR-like)
        np.random.seed(42)
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    
    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Visualizar dados
    st.subheader("📊 Visualização dos Dados")
    
    fig_data = px.scatter(
        x=X_scaled[:, 0], y=X_scaled[:, 1], 
        color=y, 
        title=f"Dataset: {dataset_type}",
        color_discrete_map={0: 'red', 1: 'blue'}
    )
    st.plotly_chart(fig_data, use_container_width=True)
    
    # Botão para treinar
    if st.button("🚀 Treinar Rede Neural", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulação de treinamento (aqui você implementaria a rede neural real)
        training_errors = []
        
        for epoch in range(epochs_pg):
            # Simulação de erro decrescente com ruído
            error = 1.0 * np.exp(-epoch * 0.02) + np.random.normal(0, 0.01)
            training_errors.append(max(0, error))
            
            # Atualizar barra de progresso
            progress = (epoch + 1) / epochs_pg
            progress_bar.progress(progress)
            status_text.text(f'Época {epoch + 1}/{epochs_pg} - Erro: {error:.4f}')
        
        # Resultados do treinamento
        st.success("✅ Treinamento Concluído!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Gráfico de convergência
            fig_conv = px.line(
                y=training_errors, 
                title="Convergência do Treinamento",
                labels={'y': 'Erro', 'index': 'Época'}
            )
            st.plotly_chart(fig_conv, use_container_width=True)
        
        with col2:
            # Métricas finais
            final_error = training_errors[-1]
            st.metric("Erro Final", f"{final_error:.4f}")
            st.metric("Épocas para Convergência", f"{len(training_errors)}")
            
            # Simulação de acurácia
            accuracy = max(0.5, 1 - final_error)
            st.metric("Acurácia Estimada", f"{accuracy:.2%}")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🧠 <strong>Redes Neurais - Guia Didático Interativo</strong><br>
    Desenvolvido para facilitar o aprendizado de conceitos fundamentais de Machine Learning
</div>
""", unsafe_allow_html=True)