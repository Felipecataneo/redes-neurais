import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.signal import convolve2d
from sklearn.model_selection import train_test_split

# Bloco try-except para o TensorFlow, para não impedir a execução se não estiver instalado
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# --- CONFIGURAÇÃO DA PÁGINA E CSS ---
st.set_page_config(
    page_title="Guia Avançado de Redes Neurais",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2.2rem;
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


# --- TÍTULO PRINCIPAL ---
st.markdown('<h1 class="main-header">🧠 Guia Didático Interativo de Redes Neurais</h1>', unsafe_allow_html=True)


# --- SIDEBAR DE NAVEGAÇÃO (CONSOLIDADA) ---
st.sidebar.title("📚 Navegação")
secao = st.sidebar.radio(
    "Escolha uma seção:",
    [
        "🔍 Introdução",
        "🔧 Perceptron",
        "🌐 Redes Neurais",
        "🔄 Backpropagation",
        "🧠 MLP em Ação",
        "🖼️ Redes Neurais Convolucionais (CNN)",
        "📜 Redes Neurais Recorrentes (RNN)",
        "🎮 Playground Interativo"
    ]
)

# --- FUNÇÕES AUXILIARES ---

# Função para criar visualização do perceptron (da sua versão original)
def plot_perceptron(weights, bias, X, y, title="Perceptron"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Arquitetura do Perceptron
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 6)
    ax1.set_title("Arquitetura do Perceptron", fontsize=14, fontweight='bold')
    for i in range(2):
        ax1.add_patch(Circle((1, 2 + i * 2), 0.3, color='lightblue', ec='black'))
        ax1.text(1, 2 + i * 2, f'x{i+1}', ha='center', va='center', fontweight='bold')
        ax1.arrow(1.3, 2 + i * 2, 3.4, 1 - i * 2, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax1.text(3, 2.5 + i * 0.5, f'w{i+1}={weights[i]:.2f}', fontsize=10, color='red')
    ax1.add_patch(Circle((5.5, 3), 0.5, color='orange', ec='black'))
    ax1.text(5.5, 3, 'Σ', ha='center', va='center', fontsize=16, fontweight='bold')
    ax1.text(5.5, 2.2, f'bias={bias:.2f}', ha='center', fontsize=10, color='blue')
    ax1.arrow(6, 3, 2, 0, head_width=0.1, head_length=0.1, fc='green', ec='green')
    ax1.add_patch(Circle((8.5, 3), 0.3, color='lightgreen', ec='black'))
    ax1.text(8.5, 3, 'y', ha='center', va='center', fontweight='bold')
    ax1.set_aspect('equal'); ax1.axis('off')
    
    # Gráfico 2: Dados e linha de decisão
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', label='Classe 0', s=50)
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', label='Classe 1', s=50)
    if weights[1] != 0:
        x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        y_line = -(weights[0] * x_line + bias) / weights[1]
        ax2.plot(x_line, y_line, 'k--', linewidth=2, label='Linha de Decisão')
    ax2.set_xlabel('Feature 1'); ax2.set_ylabel('Feature 2')
    ax2.set_title('Classificação do Perceptron'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    return fig

# Função para visualizar rede neural (da sua versão original)
def plot_neural_network(layers):
    fig, ax = plt.subplots(figsize=(12, 8))
    layer_positions = np.linspace(1, 10, len(layers))
    max_neurons = max(layers) if layers else 1
    colors = ['lightblue', 'orange', 'lightgreen']
    for i, (pos, neurons) in enumerate(zip(layer_positions, layers)):
        y_positions = np.linspace(1, max_neurons, neurons)
        color = colors[i % len(colors)]
        for j, y_pos in enumerate(y_positions):
            ax.add_patch(Circle((pos, y_pos), 0.2, color=color, ec='black'))
            if i < len(layers) - 1:
                next_y_positions = np.linspace(1, max_neurons, layers[i + 1])
                for next_y in next_y_positions:
                    ax.plot([pos + 0.2, layer_positions[i + 1] - 0.2], [y_pos, next_y], 'k-', alpha=0.3, linewidth=0.5)
    labels = ['Entrada', 'Oculta', 'Saída']
    for i, pos in enumerate(layer_positions):
        if i < len(labels):
            ax.text(pos, max_neurons + 0.5, labels[i], ha='center', fontweight='bold')
    ax.set_xlim(0, 11); ax.set_ylim(0, max_neurons + 1); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Arquitetura da Rede Neural', fontsize=16, fontweight='bold')
    return fig

# Implementação simples do perceptron (da sua versão original)
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
            if epoch_errors == 0: break
    
    def predict(self, x):
        activation = np.dot(x, self.weights) + self.bias
        return 1 if activation >= 0 else 0

# Função para plotar fronteira de decisão (das versões mais novas)
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', showscale=False, opacity=0.8))
    fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', marker_color='red', name='Classe 0'))
    fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', marker_color='blue', name='Classe 1'))
    fig.update_layout(title="Fronteira de Decisão da Rede Neural", xaxis_title="Feature 1", yaxis_title="Feature 2")
    return fig

# --- SEÇÕES DO APLICATIVO ---

if secao == "🔍 Introdução":
    st.markdown('<h2 class="section-header">O que são Redes Neurais?</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="info-box">
        As <strong>Redes Neurais Artificiais (RNAs)</strong> são modelos computacionais inspirados pela estrutura e funcionamento do cérebro humano. Elas constituem o núcleo de muitos dos avanços em Inteligência Artificial e são capazes de aprender padrões complexos a partir de dados através de um processo de treinamento.
        
        <h4>Componentes Fundamentais:</h4>
        <ul>
            <li><b>Neurônios (ou Nós):</b> As unidades computacionais básicas que recebem entradas, processam-nas e geram uma saída.</li>
            <li><b>Conexões e Pesos:</b> Cada conexão entre neurônios possui um peso associado, que modula a força do sinal. O aprendizado ocorre pelo ajuste desses pesos.</li>
            <li><b>Bias:</b> Um parâmetro extra, similar ao intercepto em uma regressão linear, que permite deslocar a função de ativação.</li>
            <li><b>Função de Ativação:</b> Determina a saída do neurônio, introduzindo não-linearidades que permitem à rede aprender padrões complexos.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("images.png", caption="Modelo de um neurônio artificial.")

# --- SEÇÃO PERCEPTRON (SUA VERSÃO ORIGINAL INTERATIVA) ---
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
        st.markdown("### 📐 Funcionamento Matemático")
        st.markdown("""
        <div class="formula-box">
        <strong>Soma Ponderada:</strong><br>
        z = w₁x₁ + w₂x₂ + ... + wₙxₙ + bias<br><br>
        <strong>Função de Ativação (Degrau):</strong><br>
        y = 1 se z ≥ 0, caso contrário y = 0
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎮 Controles Interativos")
        w1 = st.slider("Peso w₁", -2.0, 2.0, 0.5, 0.1)
        w2 = st.slider("Peso w₂", -2.0, 2.0, -0.3, 0.1)
        bias = st.slider("Bias", -2.0, 2.0, 0.0, 0.1)
        
        np.random.seed(42)
        X_example, y_example = make_classification(n_samples=50, n_features=2, n_redundant=0, 
                                                 n_informative=2, n_clusters_per_class=1, 
                                                 random_state=1)
    
    st.subheader("📊 Visualização do Perceptron (Manipulação Manual)")
    st.markdown("Use os sliders acima para ver como os pesos e o bias afetam a arquitetura e a linha de decisão.")
    fig = plot_perceptron([w1, w2], bias, X_example, y_example)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🧪 Treinamento do Perceptron")
    
    col1_train, col2_train = st.columns([1, 1])
    
    with col1_train:
        st.markdown("**Configurações de Treinamento:**")
        learning_rate = st.slider("Taxa de Aprendizado", 0.01, 1.0, 0.1, 0.01, key='lr_p')
        epochs = st.slider("Número de Épocas", 10, 200, 50, 10, key='ep_p')
        
        if st.button("🚀 Treinar Perceptron"):
            X_train, y_train = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                                 n_informative=2, n_clusters_per_class=1, 
                                                 random_state=42)
            perceptron = SimplePerceptron(learning_rate=learning_rate, epochs=epochs)
            perceptron.fit(X_train, y_train)
            st.session_state.perceptron = perceptron
            st.session_state.X_train_p = X_train
            st.session_state.y_train_p = y_train
    
    with col2_train:
        if 'perceptron' in st.session_state:
            st.markdown("**Resultados do Treinamento:**")
            perceptron = st.session_state.perceptron
            X_train = st.session_state.X_train_p
            y_train = st.session_state.y_train_p
            
            fig_conv, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
            ax.set_xlabel('Época'); ax.set_ylabel('Número de Erros')
            ax.set_title('Convergência do Perceptron'); ax.grid(True, alpha=0.3)
            st.pyplot(fig_conv)
            
            fig_result = plot_perceptron(perceptron.weights, perceptron.bias, X_train, y_train, "Resultado Final")
            st.pyplot(fig_result)

# --- SEÇÃO REDES NEURAIS (SUA VERSÃO ORIGINAL INTERATIVA) ---
elif secao == "🌐 Redes Neurais":
    st.markdown('<h2 class="section-header">Redes Neurais Multicamadas (MLP)</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="info-box">
        As <strong>Redes Neurais Multicamadas</strong> (MLPs - Multi-Layer Perceptrons) superam as limitações do perceptron simples,
        sendo capazes de resolver problemas não-linearmente separáveis através de camadas ocultas e funções de ativação não-lineares.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        ### 🏗️ Arquitetura
        - **Camada de Entrada**: Recebe os dados brutos.
        - **Camadas Ocultas**: Processam e extraem características dos dados.
        - **Camada de Saída**: Produz o resultado final.
        """)
    
    with col2:
        st.subheader("🎮 Configurar Arquitetura")
        input_neurons = st.number_input("Neurônios de Entrada", 2, 10, 3, key='nn_in')
        hidden_neurons = st.number_input("Neurônios Ocultos", 2, 20, 5, key='nn_hid')
        output_neurons = st.number_input("Neurônios de Saída", 1, 5, 1, key='nn_out')
        layers = [input_neurons, hidden_neurons, output_neurons]

    st.subheader("📊 Visualização da Arquitetura")
    st.markdown("Use os controles acima para montar a arquitetura da sua rede.")
    fig_nn = plot_neural_network(layers)
    st.pyplot(fig_nn)
    
    st.markdown("---")
    st.subheader("📈 Funções de Ativação Comuns")
    st.markdown("Essas funções introduzem não-linearidade, permitindo que a rede aprenda padrões complexos.")
    x = np.linspace(-5, 5, 100)
    sigmoid = 1 / (1 + np.exp(-x))
    relu = np.maximum(0, x)
    tanh = np.tanh(x)
    
    fig_activation, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig_activation.suptitle("Visualização de Funções de Ativação", fontsize=16)
    ax1.plot(x, sigmoid, 'b-', linewidth=2); ax1.set_title('Sigmoid'); ax1.grid(True, alpha=0.3); ax1.set_ylim(-0.1, 1.1)
    ax2.plot(x, relu, 'r-', linewidth=2); ax2.set_title('ReLU'); ax2.grid(True, alpha=0.3)
    ax3.plot(x, tanh, 'g-', linewidth=2); ax3.set_title('Tanh'); ax3.grid(True, alpha=0.3); ax3.set_ylim(-1.1, 1.1)
    ax4.plot(x, sigmoid, 'b-', label='Sigmoid'); ax4.plot(x, relu, 'r-', label='ReLU'); ax4.plot(x, tanh, 'g-', label='Tanh')
    ax4.set_title('Comparação'); ax4.legend(); ax4.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_activation)

elif secao == "🔄 Backpropagation":
    st.markdown('<h2 class="section-header">Backpropagation: Um Passo de Cada Vez</h2>', unsafe_allow_html=True)
    st.markdown("""
    O Backpropagation pode parecer uma "caixa preta". Vamos abri-la e executar um único passo de treinamento de forma interativa. 
    Veremos exatamente como a rede usa o erro para descobrir como ajustar seus pesos.
    
    Nosso cenário: uma rede com **1 neurônio**, **2 entradas**, e função de ativação **Sigmoid**.
    """)

    # --- Configuração do Exemplo Interativo ---
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("⚙️ Controles")
        learning_rate = st.slider("Taxa de Aprendizado (η)", 0.01, 1.0, 0.5, 0.01)

        # Inicializa o estado da sessão para controlar os passos
        if 'bp_step' not in st.session_state:
            st.session_state.bp_step = 0

        def next_step():
            st.session_state.bp_step += 1
        
        def reset_steps():
            st.session_state.bp_step = 0
            # Reinicia os pesos para consistência
            st.session_state.w1 = 0.3
            st.session_state.w2 = -0.5
            st.session_state.bias = 0.1

        # Inicializa os pesos se não existirem
        if 'w1' not in st.session_state:
            reset_steps()

        c1, c2 = st.columns(2)
        c1.button("Próximo Passo ➡️", on_click=next_step, type="primary", use_container_width=True)
        c2.button("Reiniciar 🔄", on_click=reset_steps, use_container_width=True)

        # --- Parâmetros do nosso exemplo ---
        x = np.array([2.0, 3.0])
        y_real = 1.0
        w = np.array([st.session_state.w1, st.session_state.w2])
        bias = st.session_state.bias

    with col2:
        st.subheader("Visualização da Descida do Gradiente")
        # Gráfico simples para ilustrar a descida
        w_space = np.linspace(-1, 1, 100)
        # Função de erro simplificada (parábola) para fins de visualização
        error_space = (w_space - 0.7)**2 
        
        fig, ax = plt.subplots()
        ax.plot(w_space, error_space, label="Superfície de Erro")
        ax.set_xlabel("Valor do Peso (Ex: w1)")
        ax.set_ylabel("Erro")
        ax.set_title("O Objetivo: Atingir o Mínimo do Erro")
        
        # Ponto inicial
        ax.plot(st.session_state.w1, (st.session_state.w1 - 0.7)**2, 'ro', markersize=10, label="Peso Atual")

        if st.session_state.bp_step >= 5:
            w_novo_calculado = w[0] - learning_rate * st.session_state.grad_w1
            ax.plot(w_novo_calculado, (w_novo_calculado - 0.7)**2, 'go', markersize=10, label="Peso Novo")
            ax.annotate("", xy=(w_novo_calculado, (w_novo_calculado-0.7)**2), xytext=(st.session_state.w1, (st.session_state.w1-0.7)**2),
                        arrowprops=dict(arrowstyle="->", color="purple", lw=2))
        
        ax.legend()
        st.pyplot(fig)


    st.markdown("---")
    st.subheader("🔍 O Processo Detalhado")

    # Passo 0: Estado Inicial
    st.markdown("##### 🏁 Estado Inicial")
    st.markdown(f"""
    - **Entradas:** `x₁ = {x[0]}`, `x₂ = {x[1]}`
    - **Alvo Real:** `y_real = {y_real}`
    - **Pesos Iniciais:** `w₁ = {w[0]:.2f}`, `w₂ = {w[1]:.2f}`
    - **Bias Inicial:** `b = {bias:.2f}`
    """)
    
    # Passo 1: Forward Pass
    if st.session_state.bp_step >= 1:
        with st.container(border=True):
            st.markdown("##### Passo 1: Forward Pass - Calcular a Predição da Rede")
            z = np.dot(w, x) + bias
            predicao = 1 / (1 + np.exp(-z)) # Função Sigmoid

            st.latex(r"z = (w₁ \times x₁) + (w₂ \times x₂) + b")
            st.markdown(f"`z = ({w[0]:.2f} × {x[0]}) + ({w[1]:.2f} × {x[1]}) + {bias:.2f} = {z:.4f}`")

            st.latex(r"\text{predição (a)} = \sigma(z) = \frac{1}{1 + e^{-z}}")
            st.markdown(f"`predição = σ({z:.4f}) = {predicao:.4f}`")
            st.session_state.predicao = predicao
            st.session_state.z = z

    # Passo 2: Calcular o Erro
    if st.session_state.bp_step >= 2:
        with st.container(border=True):
            st.markdown("##### Passo 2: Calcular o Erro da Predição")
            erro = 0.5 * (y_real - st.session_state.predicao)**2 # Metade do Erro Quadrático
            st.latex(r"E = \frac{1}{2} (y_{real} - \text{predição})^2")
            st.markdown(f"`E = 0.5 * ({y_real} - {st.session_state.predicao:.4f})² = {erro:.4f}`")

    # Passo 3: Backward Pass (Cálculo dos Gradientes)
    if st.session_state.bp_step >= 3:
        with st.container(border=True):
            st.markdown("##### Passo 3: Backward Pass - Calcular os Gradientes (A Mágica da Regra da Cadeia)")
            st.info("O objetivo é descobrir a contribuição de cada peso para o erro. Usamos a Regra da Cadeia do cálculo: `∂E/∂w = (∂E/∂predição) * (∂predição/∂z) * (∂z/∂w)`")

            # ∂E/∂predição
            dE_dpred = st.session_state.predicao - y_real
            # ∂predição/∂z (derivada da sigmoid)
            dpred_dz = st.session_state.predicao * (1 - st.session_state.predicao)
            # ∂z/∂w
            dz_dw1 = x[0]
            dz_dw2 = x[1]
            dz_dbias = 1.0

            st.markdown("**Componentes do Gradiente:**")
            st.markdown(f"- `∂E/∂predição = predição - y_real = {dE_dpred:.4f}`")
            st.markdown(f"- `∂predição/∂z = σ(z) * (1 - σ(z)) = {dpred_dz:.4f}`")
            st.markdown(f"- `∂z/∂w₁ = x₁ = {dz_dw1}`")

            st.session_state.dE_dpred = dE_dpred
            st.session_state.dpred_dz = dpred_dz

    # Passo 4: Juntar tudo para o gradiente final
    if st.session_state.bp_step >= 4:
        with st.container(border=True):
            st.markdown("##### Passo 4: Gradientes Finais")
            grad_w1 = st.session_state.dE_dpred * st.session_state.dpred_dz * x[0]
            grad_w2 = st.session_state.dE_dpred * st.session_state.dpred_dz * x[1]
            grad_bias = st.session_state.dE_dpred * st.session_state.dpred_dz * 1.0
            
            st.latex(r"\frac{\partial E}{\partial w_1} = ({dE_dpred:.2f}) \times ({st.session_state.dpred_dz:.2f}) \times ({x[0]}) = {grad_w1:.4f}")
            st.latex(r"\frac{\partial E}{\partial w_2} = ({dE_dpred:.2f}) \times ({st.session_state.dpred_dz:.2f}) \times ({x[1]}) = {grad_w2:.4f}")
            st.latex(r"\frac{\partial E}{\partial b} = ({dE_dpred:.2f}) \times ({st.session_state.dpred_dz:.2f}) \times (1) = {grad_bias:.4f}")

            # Salva para o próximo passo
            st.session_state.grad_w1 = grad_w1
            st.session_state.grad_w2 = grad_w2
            st.session_state.grad_bias = grad_bias

    # Passo 5: Atualizar os pesos
    if st.session_state.bp_step >= 5:
        with st.container(border=True):
            st.markdown("##### Passo 5: Atualizar os Pesos - O Aprendizado Acontece!")
            st.success("Finalmente, ajustamos os pesos na direção OPOSTA ao gradiente, multiplicando pela taxa de aprendizado.")
            
            w_novo1 = w[0] - learning_rate * st.session_state.grad_w1
            w_novo2 = w[1] - learning_rate * st.session_state.grad_w2
            bias_novo = bias - learning_rate * st.session_state.grad_bias

            st.latex(r"w_{novo} = w_{antigo} - \eta \times \frac{\partial E}{\partial w}")
            
            st.markdown(f"**Novos Pesos:**")
            st.markdown(f"- `w₁_novo = {w[0]:.2f} - {learning_rate} × ({st.session_state.grad_w1:.4f}) = {w_novo1:.4f}`")
            st.markdown(f"- `w₂_novo = {w[1]:.2f} - {learning_rate} × ({st.session_state.grad_w2:.4f}) = {w_novo2:.4f}`")
            st.markdown(f"- `b_novo = {bias:.2f} - {learning_rate} × ({st.session_state.grad_bias:.4f}) = {bias_novo:.4f}`")
            
            st.markdown("\nEstes seriam os novos pesos para o próximo ciclo de treinamento!")
            
    # Mensagem de fim
    if st.session_state.bp_step > 5:
        st.balloons()
        st.info("Você completou um passo de backpropagation! Clique em 'Reiniciar' para ver o processo novamente com diferentes taxas de aprendizado.")

elif secao == "🧠 MLP em Ação":
    st.markdown('<h2 class="section-header">MLP em Ação: Resolvendo Problemas Não-Lineares</h2>', unsafe_allow_html=True)
    st.markdown("""
    Vamos ver o MLP resolver um problema que o Perceptron não consegue. Para avaliar o modelo de forma realista, 
    vamos dividir nossos dados em um conjunto de **treino** (para ensinar o modelo) e um conjunto de **teste** 
    (para avaliar seu poder de generalização com dados nunca vistos).
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Parâmetros do Modelo")
        hidden_layers = st.slider("Neurônios na Camada Oculta", 2, 64, 10, key='mlp_h')
        activation = st.selectbox("Função de Ativação", ["relu", "tanh", "logistic"], key='mlp_a')
        learning_rate_mlp = st.slider("Taxa de Aprendizado", 0.001, 0.1, 0.01, format="%.3f", key='mlp_lr')
        epochs_mlp = st.slider("Máximo de Épocas", 100, 1000, 300, key='mlp_e')
        
        train_button = st.button("🚀 Treinar e Avaliar MLP", type="primary")

    # 1. Gerar os dados
    X, y = make_circles(n_samples=300, noise=0.15, factor=0.5, random_state=1)
    X_scaled = StandardScaler().fit_transform(X)
    
    # 2. DIVIDIR OS DADOS EM TREINO E TESTE
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    with col2:
        st.subheader("📊 Dados e Resultado")
        if train_button:
            model = MLPClassifier(
                hidden_layer_sizes=(hidden_layers,), 
                activation=activation, 
                solver='adam', 
                learning_rate_init=learning_rate_mlp, 
                max_iter=epochs_mlp, 
                random_state=1,
                early_stopping=True # Boa prática para evitar overfitting
            )
            
            # 3. TREINAR O MODELO APENAS COM OS DADOS DE TREINO
            model.fit(X_train, y_train)
            st.session_state.mlp_model = model
            
            # 4. AVALIAR EM AMBOS OS CONJUNTOS
            acc_train = model.score(X_train, y_train)
            acc_test = model.score(X_test, y_test)
            st.session_state.mlp_scores = (acc_train, acc_test)
            
            st.success("✅ Modelo treinado e avaliado!")

        if 'mlp_model' in st.session_state:
            # Exibe as duas métricas de acurácia
            acc_train, acc_test = st.session_state.mlp_scores
            st.metric("Acurácia no Treino (Dados Vistos)", f"{acc_train:.2%}")
            st.metric("Acurácia no Teste (Dados Novos - Generalização)", f"{acc_test:.2%}")
            
            # A fronteira de decisão é visualizada em todos os dados para um melhor entendimento
            boundary_fig = plot_decision_boundary(st.session_state.mlp_model, X_scaled, y)
            st.plotly_chart(boundary_fig, use_container_width=True)
            
            st.info("Observe como a acurácia no teste é geralmente um pouco menor que no treino. Esta é a medida mais honesta do desempenho do modelo!")
        else:
            # Antes de treinar, mostramos todos os dados para visualização
            fig_data = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], color=y.astype(str), title="Dataset 'Círculos' (Não-Linearmente Separável)")
            st.plotly_chart(fig_data, use_container_width=True)


elif secao == "🖼️ Redes Neurais Convolucionais (CNN)":
    st.markdown('<h2 class="section-header">Redes Neurais Convolucionais (CNNs)</h2>', unsafe_allow_html=True)
    # (Seção CNN completa)
    st.markdown("CNNs são uma classe de redes neurais especializada no processamento de dados com uma topologia de grade, como imagens.")
    with st.expander("👁️ A Operação de Convolução", expanded=True):
        st.markdown("A convolução aplica um **filtro (ou kernel)** sobre a imagem, criando um **mapa de características** que destaca padrões como bordas ou texturas.")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Filtros Interativos")
            image_data = np.zeros((10, 10)); image_data[2:8, 2:8] = 10
            kernels = {"Detector de Borda Vertical": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), "Detector de Borda Horizontal": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])}
            kernel_choice = st.selectbox("Escolha um Kernel:", list(kernels.keys()))
            kernel = kernels[kernel_choice]
        with col2:
            convolved_image = convolve2d(image_data, kernel, mode='valid')
            fig, ax = plt.subplots(1, 3, figsize=(12, 4)); ax[0].imshow(image_data, cmap='gray'); ax[0].set_title('Imagem de Entrada'); ax[1].imshow(kernel, cmap='gray'); ax[1].set_title('Kernel'); ax[2].imshow(convolved_image, cmap='gray'); ax[2].set_title('Mapa de Características')
            for a in ax: a.axis('off')
            st.pyplot(fig)

elif secao == "📜 Redes Neurais Recorrentes (RNN)":
    st.markdown('<h2 class="section-header">Redes Neurais Recorrentes (RNNs)</h2>', unsafe_allow_html=True)
    # (Seção RNN completa)
    st.markdown("RNNs são projetadas para trabalhar com dados sequenciais. Sua característica definidora é a **conexão recorrente**, que cria uma **memória** para reter informações sobre o passado.")
    st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png", caption="Uma RNN 'desdobrada' no tempo. Fonte: Chris Olah's Blog.")
    with st.expander("🧠 Solução para Memória de Longo Prazo: LSTM e GRU", expanded=True):
        st.markdown("**Long Short-Term Memory (LSTM)** e **Gated Recurrent Unit (GRU)** usam 'portões' (gates) para regular o fluxo de informação, permitindo que a rede aprenda a reter ou descartar informações de forma seletiva.")
        st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png", caption="A estrutura de uma célula LSTM com seus portões. Fonte: Chris Olah's Blog.")


elif secao == "🎮 Playground Interativo":
    st.markdown('<h2 class="section-header">Playground Interativo de MLP</h2>', unsafe_allow_html=True)
    # (Seção Playground completa e funcional)
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.subheader("🎯 Dados")
        dataset_type = st.selectbox("Tipo de Dataset", ["Círculos", "Luas", "Linearmente Separável"])
        n_samples = st.slider("Amostras", 100, 1000, 300)
    with col2:
        st.subheader("🏗️ Arquitetura")
        hl_1 = st.slider("Neurônios Camada 1", 1, 50, 10)
        hl_2 = st.slider("Neurônios Camada 2", 0, 50, 5) # 0 para desativar
        activation_pg = st.selectbox("Ativação", ["relu", "tanh"], key='pg_act')
    if dataset_type == "Linearmente Separável": X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
    elif dataset_type == "Círculos": X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    else: X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    with col3:
        st.subheader("📊 Dados de Entrada")
        fig_data = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], color=y.astype(str)); fig_data.update_layout(showlegend=False, title=f"Dataset: {dataset_type}")
        st.plotly_chart(fig_data, use_container_width=True)
    if st.button("🚀 TREINAR REDE NEURAL NO PLAYGROUND", type="primary"):
        hidden_layers_pg = (hl_1,) if hl_2 == 0 else (hl_1, hl_2)
        model_pg = MLPClassifier(hidden_layer_sizes=hidden_layers_pg, activation=activation_pg, solver='adam', max_iter=500, random_state=1, early_stopping=True, n_iter_no_change=20)
        with st.spinner("Treinando o modelo..."): model_pg.fit(X_scaled, y)
        st.session_state.pg_model = model_pg
        st.success("✅ Treinamento Concluído!")
    if 'pg_model' in st.session_state:
        res_col1, res_col2 = st.columns(2)
        model = st.session_state.pg_model
        with res_col1:
            st.markdown("#### Fronteira de Decisão"); boundary_fig = plot_decision_boundary(model, X_scaled, y); st.plotly_chart(boundary_fig, use_container_width=True)
        with res_col2:
            st.markdown("#### Métricas"); st.metric("Acurácia no Treino", f"{model.score(X_scaled, y):.2%}"); st.metric("Iterações", f"{model.n_iter_}"); st.metric("Perda Final", f"{model.loss_:.4f}")


# --- RODAPÉ ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><strong>Guia Avançado de Redes Neurais</strong> | Desenvolvido para o aprendizado aprofundado de conceitos de Machine Learning.</div>", unsafe_allow_html=True)