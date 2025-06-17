import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.signal import convolve2d
import tensorflow as tf

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
    .code-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# --- TÍTULO PRINCIPAL ---
st.markdown('<h1 class="main-header">🧠 Guia Avançado de Redes Neurais</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Uma abordagem didática e interativa para estudantes de Pós-Graduação.</p>", unsafe_allow_html=True)


# --- SIDEBAR DE NAVEGAÇÃO ---
st.sidebar.title("📚 Navegação")
secao = st.sidebar.selectbox(
    "Escolha um tópico:",
    [
        "🔍 Introdução",
        "🔧 O Perceptron",
        "🏛️ Arquitetura de Redes Neurais",
        "🧠 Multilayer Perceptron (MLP) em Ação",
        "🔄 Backpropagation",
        "🖼️ Redes Neurais Convolucionais (CNN)",
        "📜 Redes Neurais Recorrentes (RNN)",
        "🎮 Playground Interativo"
    ]
)

# --- FUNÇÕES AUXILIARES ---

# Função para plotar o Perceptron (mantida do original)
def plot_perceptron(weights, bias, X, y, title="Perceptron"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Arquitetura
    ax[0].set_xlim(0, 10); ax[0].set_ylim(0, 6)
    ax[0].set_title("Arquitetura do Perceptron", fontsize=14, fontweight='bold')
    for i in range(2):
        ax[0].add_patch(Circle((1, 2 + i * 2), 0.3, color='lightblue', ec='black'))
        ax[0].text(1, 2 + i * 2, f'x{i+1}', ha='center', va='center', fontweight='bold')
        ax[0].arrow(1.3, 2 + i * 2, 3.4, 1 - i * 2, head_width=0.1, fc='red', ec='red')
        ax[0].text(3, 2.5 + i * 0.5, f'w{i+1}={weights[i]:.2f}', fontsize=10, color='red')
    ax[0].add_patch(Circle((5.5, 3), 0.5, color='orange', ec='black'))
    ax[0].text(5.5, 3, 'Σ', ha='center', va='center', fontsize=16, fontweight='bold')
    ax[0].text(5.5, 2.2, f'bias={bias:.2f}', ha='center', fontsize=10, color='blue')
    ax[0].arrow(6, 3, 2, 0, head_width=0.1, fc='green', ec='green')
    ax[0].add_patch(Circle((8.5, 3), 0.3, color='lightgreen', ec='black'))
    ax[0].text(8.5, 3, 'y', ha='center', va='center', fontweight='bold')
    ax[0].axis('off')

    # Gráfico 2: Dados e linha de decisão
    ax[1].scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', label='Classe 0')
    ax[1].scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', label='Classe 1')
    if weights[1] != 0:
        x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        y_line = -(weights[0] * x_line + bias) / weights[1]
        ax[1].plot(x_line, y_line, 'k--', linewidth=2, label='Fronteira de Decisão')
    ax[1].set_xlabel('Feature 1'); ax[1].set_ylabel('Feature 2')
    ax[1].set_title('Classificação do Perceptron'); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    
    return fig

# Função para visualizar arquitetura de rede neural (mantida)
def plot_neural_network(layers):
    fig, ax = plt.subplots(figsize=(12, 8))
    layer_positions = np.linspace(1, 10, len(layers))
    max_neurons = max(layers)
    for i, (pos, neurons) in enumerate(zip(layer_positions, layers)):
        y_positions = np.linspace(1, max_neurons, neurons)
        for j, y_pos in enumerate(y_positions):
            ax.add_patch(Circle((pos, y_pos), 0.2, color='orange', ec='black'))
            if i < len(layers) - 1:
                next_y_positions = np.linspace(1, max_neurons, layers[i + 1])
                for next_y in next_y_positions:
                    ax.plot([pos + 0.2, layer_positions[i + 1] - 0.2], 
                           [y_pos, next_y], 'k-', alpha=0.3)
    labels = ['Entrada', 'Oculta', 'Saída']
    for i, pos in enumerate(layer_positions):
        if i < len(labels):
            ax.text(pos, max_neurons + 0.5, labels[i], ha='center', fontweight='bold')
    ax.set_xlim(0, 11); ax.set_ylim(0, max_neurons + 1); ax.axis('off')
    ax.set_title('Arquitetura da Rede Neural', fontsize=16, fontweight='bold')
    return fig

# Função para plotar fronteira de decisão (NOVA)
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

# Implementação do Perceptron (mantida do original)
class SimplePerceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr, self.epochs = learning_rate, epochs
        self.weights, self.bias, self.errors = None, None, []
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features); self.bias = 0; self.errors = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            self.errors.append(errors)
            if errors == 0: break
    def predict(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0.0 else 0

# --- SEÇÕES DO APLICATIVO ---

if secao == "🔍 Introdução":
    # (Código da Introdução original, sem alterações)
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
        st.image("https://www.google.com/imgres?q=Artificial%20Neuron.png&imgurl=https%3A%2F%2Fimg.favpng.com%2F14%2F13%2F19%2Fartificial-neural-network-deep-learning-artificial-intelligence-machine-learning-neuron-png-favpng-R2h2Atu1ukdBmjhEyMzXmVtPD.jpg&imgrefurl=https%3A%2F%2Ffavpng.com%2Fpng_view%2Fbrain-artificial-neural-network-deep-learning-artificial-intelligence-machine-learning-neuron-png%2FkR6FGUUd&docid=xENqRVzuapGZeM&tbnid=kPEG4siK8nit4M&vet=12ahUKEwjqkseV2_iNAxUQq5UCHQNxG6YQM3oECHMQAA..i&w=820&h=440&hcb=2&ved=2ahUKEwjqkseV2_iNAxUQq5UCHQNxG6YQM3oECHMQAA", 
                 caption="Modelo de um neurônio artificial.")
    
elif secao == "🔧 O Perceptron":
    # (Código do Perceptron original, sem alterações significativas)
    st.markdown('<h2 class="section-header">O Perceptron: O Bloco de Construção Fundamental</h2>', unsafe_allow_html=True)
    st.markdown("""O Perceptron, concebido por Frank Rosenblatt em 1957, é o modelo mais simples de um neurônio artificial. Ele serve como um classificador binário linear, o que significa que pode aprender a separar dados que são linearmente separáveis.""")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>Funcionamento Matemático</h4>
        1. <strong>Soma Ponderada (z):</strong> As entradas (x) são multiplicadas pelos seus respectivos pesos (w) e somadas, junto com o bias (b).<br>
        2. <strong>Função de Ativação:</strong> Uma função degrau (Heaviside) é aplicada à soma ponderada. Se `z` for maior ou igual a um limiar (geralmente 0), a saída é 1; caso contrário, é 0.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
        z = (∑ wᵢxᵢ) + b <br>
        y = { 1 se z ≥ 0; 0 se z < 0 }
        </div>
        """, unsafe_allow_html=True)
        st.warning("**Limitação Crítica:** O Perceptron só pode resolver problemas linearmente separáveis. Ele falha em problemas como o XOR.", icon="⚠️")

    with col2:
        st.subheader("🧪 Demonstração de Treinamento")
        if st.button("🚀 Treinar Perceptron em Dados Lineares"):
            X_train, y_train = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                                 n_informative=2, n_clusters_per_class=1, 
                                                 flip_y=0, random_state=1)
            perceptron = SimplePerceptron(learning_rate=0.1, epochs=100)
            perceptron.fit(X_train, y_train)
            st.session_state.perceptron_trained = perceptron
            st.session_state.perceptron_data = (X_train, y_train)

    if 'perceptron_trained' in st.session_state:
        perceptron = st.session_state.perceptron_trained
        X_train, y_train = st.session_state.perceptron_data
        
        fig_result = plot_perceptron(perceptron.weights, perceptron.bias, X_train, y_train, "Resultado do Treinamento")
        st.pyplot(fig_result)
        
        st.markdown("A linha tracejada representa a fronteira de decisão que o Perceptron aprendeu para separar as duas classes.")

elif secao == "🏛️ Arquitetura de Redes Neurais":
    # (Código da seção de Redes Neurais, renomeada para focar na arquitetura)
    st.markdown('<h2 class="section-header">Arquitetura de Redes Neurais Multicamadas (MLP)</h2>', unsafe_allow_html=True)
    st.markdown("""
    Para superar a limitação do Perceptron, combinamos múltiplos neurônios em camadas. Uma **Rede Neural Multicamadas (MLP)** consiste em:
    - Uma **Camada de Entrada** que recebe os dados.
    - Uma ou mais **Camadas Ocultas** que realizam o processamento intermediário e a extração de características. É aqui que a "mágica" acontece.
    - Uma **Camada de Saída** que produz o resultado final (e.g., uma probabilidade de classe).

    A presença de camadas ocultas e funções de ativação não-lineares permite que os MLPs aprendam fronteiras de decisão complexas e não-lineares.
    """)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Visualização da Arquitetura")
        layers = [
            st.session_state.get('input_neurons', 4), 
            st.session_state.get('hidden_neurons', 8), 
            st.session_state.get('hidden_neurons_2', 5), 
            st.session_state.get('output_neurons', 2)
        ]
        fig_nn = plot_neural_network(layers)
        st.pyplot(fig_nn)
    with col2:
        st.subheader("🎮 Configurar Arquitetura")
        st.session_state.input_neurons = st.slider("Neurônios de Entrada", 2, 10, 4)
        st.session_state.hidden_neurons = st.slider("Neurônios na Camada Oculta 1", 2, 20, 8)
        st.session_state.hidden_neurons_2 = st.slider("Neurônios na Camada Oculta 2", 2, 20, 5)
        st.session_state.output_neurons = st.slider("Neurônios de Saída", 1, 5, 2)
    
    st.subheader("📈 Funções de Ativação Não-Lineares")
    st.markdown("Funções de ativação como Sigmoid, Tanh e, especialmente, **ReLU (Rectified Linear Unit)**, são essenciais. A ReLU é a mais popular atualmente devido à sua simplicidade e eficácia em mitigar o problema do desaparecimento do gradiente.")
    x_act = np.linspace(-5, 5, 100)
    df_act = pd.DataFrame({
        'x': x_act,
        'ReLU': np.maximum(0, x_act),
        'Sigmoid': 1 / (1 + np.exp(-x_act)),
        'Tanh': np.tanh(x_act)
    })
    fig_act = px.line(df_act, x='x', y=['ReLU', 'Sigmoid', 'Tanh'], title="Funções de Ativação Comuns")
    st.plotly_chart(fig_act, use_container_width=True)

# --- NOVA SEÇÃO: MLP em Ação ---
elif secao == "🧠 Multilayer Perceptron (MLP) em Ação":
    st.markdown('<h2 class="section-header">MLP em Ação: Resolvendo Problemas Não-Lineares</h2>', unsafe_allow_html=True)
    st.markdown("""
    Vamos ver o MLP resolver um problema que o Perceptron não consegue: classificar dados que formam círculos concêntricos.
    Usaremos a implementação `MLPClassifier` da biblioteca `scikit-learn` para demonstrar o poder das camadas ocultas.
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Parâmetros do Modelo")
        hidden_layers = st.slider("Neurônios na Camada Oculta", 2, 64, 10, key='mlp_h')
        activation = st.selectbox("Função de Ativação", ["relu", "tanh", "logistic"], key='mlp_a')
        learning_rate = st.slider("Taxa de Aprendizado", 0.001, 0.1, 0.01, format="%.3f", key='mlp_lr')
        epochs = st.slider("Máximo de Épocas", 100, 1000, 300, key='mlp_e')
        
        train_button = st.button("🚀 Treinar MLP", type="primary")

    # Gerar dados
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=1)
    X_scaled = StandardScaler().fit_transform(X)

    with col2:
        st.subheader("📊 Dados e Resultado")
        if train_button:
            model = MLPClassifier(
                hidden_layer_sizes=(hidden_layers,),
                activation=activation,
                solver='adam',
                learning_rate_init=learning_rate,
                max_iter=epochs,
                random_state=1
            )
            model.fit(X_scaled, y)
            st.session_state.mlp_model = model
            
            # Plotar curva de perda
            loss_fig = px.line(y=model.loss_curve_, labels={'x': 'Época', 'y': 'Perda'}, title="Curva de Perda do Treinamento")
            st.plotly_chart(loss_fig, use_container_width=True)
        
        if 'mlp_model' in st.session_state:
            # Plotar fronteira de decisão
            boundary_fig = plot_decision_boundary(st.session_state.mlp_model, X_scaled, y)
            st.plotly_chart(boundary_fig, use_container_width=True)
            st.info(f"Acurácia do modelo: {st.session_state.mlp_model.score(X_scaled, y):.2%}")
        else:
            fig_data = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], color=y.astype(str), title="Dataset 'Círculos' (Não-Linearmente Separável)")
            st.plotly_chart(fig_data, use_container_width=True)
            st.info("Clique em 'Treinar MLP' para ver o resultado.")

elif secao == "🔄 Backpropagation":
    # (Código da seção de Backpropagation, sem alterações significativas)
    st.markdown('<h2 class="section-header">Backpropagation: O Algoritmo de Aprendizado</h2>', unsafe_allow_html=True)
    st.markdown("""
    O **Backpropagation** (retropropagação do erro) é o algoritmo que permite que as redes neurais aprendam. Ele funciona em conjunto com um método de otimização, como o **Gradiente Descendente**.
    A ideia central é calcular o gradiente da função de perda (erro) em relação a cada peso e bias da rede. Este gradiente aponta a direção de maior crescimento do erro, então ajustamos os pesos na direção *oposta* para minimizá-lo.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>Processo Simplificado:</h4>
        <ol>
            <li><b>Forward Pass:</b> Os dados de entrada são passados pela rede para gerar uma predição.</li>
            <li><b>Cálculo da Perda:</b> A predição é comparada com o valor real para calcular o erro (e.g., erro quadrático médio).</li>
            <li><b>Backward Pass:</b> A "mágica" acontece aqui. Usando a <b>Regra da Cadeia</b> do cálculo, o erro é propagado de volta, da camada de saída para a de entrada, calculando a contribuição de cada peso para o erro total.</li>
            <li><b>Atualização dos Pesos:</b> Os pesos são atualizados na direção oposta ao seu gradiente, diminuindo o erro.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="formula-box">
        <b>Regra da Cadeia (conceitual):</b><br>
        <sup>∂Erro</sup>⁄<sub>∂peso</sub> = <sup>∂Erro</sup>⁄<sub>∂saída</sub> × <sup>∂saída</sup>⁄<sub>∂soma</sub> × <sup>∂soma</sup>⁄<sub>∂peso</sub>
        <br><br>
        <b>Atualização do Peso (Gradiente Descendente):</b><br>
        peso<sub>novo</sub> = peso<sub>antigo</sub> - η × <sup>∂Erro</sup>⁄<sub>∂peso</sub>
        <br><i>(η é a taxa de aprendizado)</i>
        </div>
        """, unsafe_allow_html=True)

# --- NOVA SEÇÃO: CNN ---
elif secao == "🖼️ Redes Neurais Convolucionais (CNN)":
    st.markdown('<h2 class="section-header">Redes Neurais Convolucionais (CNNs)</h2>', unsafe_allow_html=True)
    st.markdown("""
    CNNs são uma classe de redes neurais especializada no processamento de dados com uma topologia de grade, como imagens. Elas utilizam duas operações fundamentais: **convolução** e **pooling** para extrair hierarquias de características espaciais.
    """)
    
    with st.expander("👁️ A Operação de Convolução", expanded=True):
        st.markdown("""
        A convolução aplica um **filtro (ou kernel)** sobre a imagem de entrada. O filtro é uma pequena matriz de pesos que desliza sobre a imagem, computando o produto de ponto em cada posição. O resultado é um **mapa de características (feature map)**, que destaca padrões específicos, como bordas, texturas ou formas.
        """)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Filtros Interativos")
            image = np.array([
                [0, 0, 0, 10, 10, 10, 0, 0, 0],
                [0, 0, 0, 10, 10, 10, 0, 0, 0],
                [0, 0, 0, 10, 10, 10, 0, 0, 0],
                [0, 0, 0, 10, 10, 10, 0, 0, 0],
                [0, 0, 0, 10, 0, 10, 0, 0, 0],
                [0, 0, 0, 10, 0, 10, 0, 0, 0],
                [0, 0, 0, 10, 0, 10, 0, 0, 0],
            ])

            kernels = {
                "Detector de Borda Vertical": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
                "Detector de Borda Horizontal": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
                "Sharpen (Aguçar)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            }
            kernel_choice = st.selectbox("Escolha um Kernel:", list(kernels.keys()))
            kernel = kernels[kernel_choice]
            
        with col2:
            convolved_image = convolve2d(image, kernel, mode='valid')
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(image, cmap='gray'); ax[0].set_title('Imagem de Entrada')
            ax[1].imshow(kernel, cmap='gray'); ax[1].set_title('Kernel')
            ax[2].imshow(convolved_image, cmap='gray'); ax[2].set_title('Mapa de Características')
            for a in ax: a.axis('off')
            st.pyplot(fig)

    with st.expander("📉 A Operação de Pooling", expanded=False):
        st.markdown("""
        O pooling (geralmente **Max Pooling**) reduz a dimensionalidade espacial dos mapas de características. Ele opera em uma janela (e.g., 2x2) e seleciona o valor máximo, tornando a representação mais robusta a pequenas translações e reduzindo a carga computacional.
        """)
        input_map = np.array([[1, 4, 2, 8], [9, 3, 5, 7], [2, 6, 1, 0], [8, 4, 3, 5]])
        output_map = np.array([
            [np.max(input_map[0:2, 0:2]), np.max(input_map[0:2, 2:4])],
            [np.max(input_map[2:4, 0:2]), np.max(input_map[2:4, 2:4])]
        ])
        
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(input_map, cmap='viridis'); ax[0].set_title("Antes do Pooling (4x4)")
        ax[1].imshow(output_map, cmap='viridis'); ax[1].set_title("Depois do Max Pooling 2x2 (2x2)")
        for a in ax:
            for i in range(a.get_images()[0].get_array().shape[0]):
                for j in range(a.get_images()[0].get_array().shape[1]):
                    a.text(j, i, a.get_images()[0].get_array()[i, j], ha="center", va="center", color="w")
        st.pyplot(fig)

    st.subheader("🏗️ Arquitetura Típica de uma CNN")
    st.markdown("Uma CNN empilha camadas de `Convolução -> Ativação (ReLU) -> Pooling` múltiplas vezes, seguidas por camadas totalmente conectadas (como um MLP) para a classificação final.")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*uAeANQIOuSKOLFiJIhNB3A.png", caption="Exemplo de arquitetura de CNN para classificação de imagens (LeNet-5).")
    st.code("""
# Exemplo de código de uma CNN simples com TensorFlow/Keras
model = tf.keras.models.Sequential([
    # Camada convolucional
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    # Camada de pooling
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Achatamento para a camada densa
    tf.keras.layers.Flatten(),
    
    # Camada densa (MLP)
    tf.keras.layers.Dense(128, activation='relu'),
    # Camada de saída
    tf.keras.layers.Dense(10, activation='softmax')
])
""", language='python')

# --- NOVA SEÇÃO: RNN ---
elif secao == "📜 Redes Neurais Recorrentes (RNN)":
    st.markdown('<h2 class="section-header">Redes Neurais Recorrentes (RNNs)</h2>', unsafe_allow_html=True)
    st.markdown("""
    RNNs são projetadas para trabalhar com dados sequenciais, como séries temporais, texto ou áudio. Sua característica definidora é a **conexão recorrente**: a saída de um neurônio em um passo de tempo é alimentada de volta para si mesmo no próximo passo de tempo. Isso cria um "estado oculto" que atua como uma **memória**, permitindo que a rede retenha informações sobre o passado.
    """)
    
    st.subheader("🔄 Visualizando a Recorrência")
    st.markdown("Uma RNN pode ser 'desdobrada' no tempo, revelando como a informação flui de um passo para o outro.")
    st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png", caption="Uma RNN 'enrolada' (esquerda) e 'desdobrada' no tempo (direita). Fonte: Chris Olah's Blog.")

    st.warning("⚠️ **Problema do Gradiente:** RNNs simples sofrem com o desaparecimento ou explosão do gradiente em sequências longas, dificultando o aprendizado de dependências de longo prazo.")
    
    with st.expander("🧠 Solução: LSTM e GRU", expanded=True):
        st.markdown("""
        **Long Short-Term Memory (LSTM)** e **Gated Recurrent Unit (GRU)** são tipos avançados de RNNs projetados para resolver esse problema. Elas usam "portões" (gates) — mecanismos de redes neurais que regulam o fluxo de informação.
        - **Forget Gate:** Decide qual informação do estado anterior deve ser descartada.
        - **Input Gate:** Decide qual nova informação deve ser armazenada.
        - **Output Gate:** Decide o que será produzido como saída.
        
        Esses portões permitem que a rede aprenda a reter informações relevantes por longos períodos e a descartar o que não é importante.
        """)
        st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png", caption="A estrutura de uma célula LSTM com seus portões. Fonte: Chris Olah's Blog.")
    
    st.subheader("🏗️ Exemplo de Arquitetura para Análise de Sentimentos")
    st.markdown("Uma arquitetura comum para classificação de texto envolve uma camada de Embedding (para converter palavras em vetores), seguida por uma camada LSTM e, finalmente, uma camada densa para a saída.")
    st.code("""
# Exemplo de código de um modelo LSTM para classificação de texto
model = tf.keras.models.Sequential([
    # Converte palavras (inteiros) em vetores densos
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64),
    
    # A camada LSTM que processa a sequência de vetores
    tf.keras.layers.LSTM(64),
    
    # Camada densa para classificação
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Saída binária (positivo/negativo)
])
""", language='python')

# --- SEÇÃO: PLAYGROUND INTERATIVO (MELHORADO) ---
elif secao == "🎮 Playground Interativo":
    st.markdown('<h2 class="section-header">Playground Interativo de MLP</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Experimente diferentes configurações e veja como elas afetam o desempenho e a fronteira de decisão de uma rede neural em tempo real!
    </div>
    """, unsafe_allow_html=True)
    
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
        
    # Gerar dados
    if dataset_type == "Linearmente Separável":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
    elif dataset_type == "Círculos":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    else: # Luas
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    
    X_scaled = StandardScaler().fit_transform(X)
    
    with col3:
        st.subheader("📊 Dados de Entrada")
        fig_data = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], color=y.astype(str))
        fig_data.update_layout(showlegend=False, title=f"Dataset: {dataset_type}")
        st.plotly_chart(fig_data, use_container_width=True)

    st.markdown("---")
    st.subheader("⚙️ Treinamento e Resultados")
    
    if st.button("🚀 TREINAR REDE NEURAL", type="primary"):
        hidden_layers_pg = (hl_1,) if hl_2 == 0 else (hl_1, hl_2)
        
        model_pg = MLPClassifier(
            hidden_layer_sizes=hidden_layers_pg,
            activation=activation_pg,
            solver='adam',
            max_iter=500,
            random_state=1,
            early_stopping=True,
            n_iter_no_change=20
        )
        
        with st.spinner("Treinando o modelo... Isso pode levar alguns segundos."):
            model_pg.fit(X_scaled, y)
        
        st.session_state.pg_model = model_pg
        st.success("✅ Treinamento Concluído!")

    if 'pg_model' in st.session_state:
        res_col1, res_col2 = st.columns(2)
        model = st.session_state.pg_model
        
        with res_col1:
            st.markdown("#### Fronteira de Decisão")
            boundary_fig = plot_decision_boundary(model, X_scaled, y)
            st.plotly_chart(boundary_fig, use_container_width=True)

        with res_col2:
            st.markdown("#### Métricas do Treinamento")
            accuracy = model.score(X_scaled, y)
            st.metric("Acurácia no Treino", f"{accuracy:.2%}")
            st.metric("Número de Iterações", f"{model.n_iter_}")
            st.metric("Perda Final", f"{model.loss_:.4f}")
            
            loss_fig = px.line(y=model.loss_curve_, labels={'x': 'Época', 'y': 'Perda'}, title="Curva de Perda")
            st.plotly_chart(loss_fig, use_container_width=True)

# --- RODAPÉ ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><strong>Guia Avançado de Redes Neurais</strong> | Desenvolvido para o aprendizado aprofundado de conceitos de Machine Learning.</div>", unsafe_allow_html=True)