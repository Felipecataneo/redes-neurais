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
from typing import List, Tuple
import time

# --- DEPENDÊNCIAS OPCIONAIS ---
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import seaborn as sns
    from transformers import AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- FUNÇÃO PARA CARREGAR TOKENIZER DA HUGGING FACE ---
@st.cache_resource
def carregar_tokenizer_hf(modelo_nome: str):
    """Baixa e carrega um tokenizer da Hugging Face."""
    try:
        return AutoTokenizer.from_pretrained(modelo_nome)
    except Exception as e:
        st.error(f"Falha ao carregar o tokenizer '{modelo_nome}': {e}")
        return None

# --- CLASSES E LÓGICA DA SEÇÃO TRANSFORMERS ---
if TORCH_AVAILABLE:
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, num_heads: int):
            super().__init__(); self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads
            self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))
        def forward(self, x: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
            bs, sl, dm = x.shape
            Q, K, V = (w(x).view(bs, sl, self.num_heads, self.d_k).transpose(1, 2) for w in (self.W_q, self.W_k, self.W_v))
            out, weights = self.scaled_dot_product_attention(Q, K, V, mask)
            out = out.transpose(1, 2).contiguous().view(bs, sl, dm)
            return self.W_o(out), weights
        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
            if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1)
            return torch.matmul(weights, V), weights

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
            super().__init__(); self.attention = MultiHeadAttention(d_model, num_heads)
            self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
            self.dropout = nn.Dropout(dropout)
        def forward(self, x: torch.Tensor, mask=None):
            attn_output, attn_weights = self.attention(x, mask)
            x = self.norm1(x + self.dropout(attn_output)); ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_output))
            return x, attn_weights

    class GPTSimplificado(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
            self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_model * 4) for _ in range(num_layers)])
            self.ln_final = nn.LayerNorm(d_model); self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        def forward(self, input_ids: torch.Tensor):
            seq_len = input_ids.shape[1]; token_embeds = self.token_embedding(input_ids)
            pos_embeds = self.position_embedding[:seq_len]; x = token_embeds + pos_embeds
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            attention_weights_list = []
            for block in self.transformer_blocks:
                x, attn_weights = block(x, mask); attention_weights_list.append(attn_weights)
            x = self.ln_final(x); logits = self.lm_head(x)
            return logits, attention_weights_list

    class DemonstradorGPT:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.model = GPTSimplificado(vocab_size=tokenizer.vocab_size, d_model=64, num_heads=4, num_layers=2, max_seq_len=50)
            self._init_weights()
            self.initial_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}

        def _init_weights(self):
            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        def reset_model(self):
            self.model.load_state_dict(self.initial_state_dict)

        def gerar_texto(self, prompt: str, max_tokens: int = 5, temperature: float = 0.7):
            self.model.eval()
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            with torch.no_grad():
                for _ in range(max_tokens):
                    outputs = self.model(input_ids); logits = outputs[0]
                    next_token_logits = logits[:, -1, :] / temperature
                    next_token_id = torch.argmax(next_token_logits, dim=-1).view(1, 1)
                    input_ids = torch.cat([input_ids, next_token_id], dim=1)
            return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        def simular_treinamento(self, st_log_placeholder):
            self.model.train()
            frases_treino = ["o gato subiu no telhado", "inteligencia artificial vai mudar o mundo", "qual e o seu modelo de linguagem favorito", "o brasil e um pais de muitas belezas"]
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
            log_text = "🏋️ Iniciando treinamento simulado com frases em português...\n"
            st_log_placeholder.markdown(f"```\n{log_text}\n```")
            for epoca in range(50):
                total_loss = 0
                for frase in frases_treino:
                    tokens = self.tokenizer.encode(frase)
                    if len(tokens) < 2: continue
                    input_ids = torch.tensor([tokens[:-1]]); targets = torch.tensor([tokens[1:]])
                    optimizer.zero_grad()
                    logits, _ = self.model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss.backward(); optimizer.step(); total_loss += loss.item()
                if (epoca + 1) % 10 == 0:
                    log_text += f"  Época {epoca+1}/50 - Perda (Loss): {total_loss:.3f}\n"
                    st_log_placeholder.markdown(f"```\n{log_text}\n```")
                    time.sleep(0.2)
            log_text += "✅ Treinamento concluído!"
            st_log_placeholder.markdown(f"```\n{log_text}\n```")

        def visualizar_atencao(self, attn_weights: np.ndarray, tokens_str: List[str]):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attn_weights, xticklabels=tokens_str, yticklabels=tokens_str, annot=True, fmt='.2f', cmap='Blues', ax=ax)
            ax.set_title('Matriz de Atenção (1ª Camada, 1ª Cabeça)', fontsize=14)
            ax.set_xlabel('Tokens (Keys - Fonte da Informação)'); ax.set_ylabel('Tokens (Queries - Buscando Informação)')
            plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
            plt.tight_layout()
            return fig

# --- CONFIGURAÇÃO DA PÁGINA E CSS ---
st.set_page_config(page_title="Guia Didático de Redes Neurais", page_icon="🧠", layout="wide")
st.markdown("""<style>.main-header{font-size:3rem;color:#1f77b4;text-align:center;margin-bottom:2rem}.section-header{font-size:2.2rem;color:#ff7f0e;border-bottom:2px solid #ff7f0e;padding-bottom:.5rem;margin:2rem 0 1rem}.info-box{background-color:#f0f8ff;padding:1rem;border-radius:10px;border-left:5px solid #1f77b4;margin:1rem 0}.formula-box{background-color:#fff8dc;padding:1rem;border-radius:10px;border:2px solid #ffa500;margin:1rem 0;text-align:center}</style>""", unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🧠 Guia Didático Interativo de Redes Neurais</h1>', unsafe_allow_html=True)
st.sidebar.title("📚 Navegação")
secao = st.sidebar.radio("Escolha uma seção:", ["🔍 Introdução", "🔧 Perceptron", "🌐 Redes Neurais", "🔄 Backpropagation", "🧠 MLP em Ação", "🖼️ CNNs", "📜 RNNs", "🤖 Transformers", "🎮 Playground Interativo"])

# --- FUNÇÕES AUXILIARES ---
def plot_perceptron(weights, bias, X, y, title="Perceptron"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 6); ax1.set_title("Arquitetura do Perceptron", fontsize=14, fontweight='bold')
    for i in range(2):
        ax1.add_patch(Circle((1, 2 + i * 2), 0.3, color='lightblue', ec='black')); ax1.text(1, 2 + i * 2, f'x{i+1}', ha='center', va='center', fontweight='bold')
        ax1.arrow(1.3, 2 + i * 2, 3.4, 1 - i * 2, head_width=0.1, head_length=0.1, fc='red', ec='red'); ax1.text(3, 2.5 + i * 0.5, f'w{i+1}={weights[i]:.2f}', fontsize=10, color='red')
    ax1.add_patch(Circle((5.5, 3), 0.5, color='orange', ec='black')); ax1.text(5.5, 3, 'Σ', ha='center', va='center', fontsize=16, fontweight='bold')
    ax1.text(5.5, 2.2, f'bias={bias:.2f}', ha='center', fontsize=10, color='blue'); ax1.arrow(6, 3, 2, 0, head_width=0.1, head_length=0.1, fc='green', ec='green')
    ax1.add_patch(Circle((8.5, 3), 0.3, color='lightgreen', ec='black')); ax1.text(8.5, 3, 'y', ha='center', va='center', fontweight='bold')
    ax1.set_aspect('equal'); ax1.axis('off')
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', label='Classe 0', s=50); ax2.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', label='Classe 1', s=50)
    if weights[1] != 0:
        x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100); y_line = -(weights[0] * x_line + bias) / weights[1]
        ax2.plot(x_line, y_line, 'k--', linewidth=2, label='Linha de Decisão')
    ax2.set_xlabel('Feature 1'); ax2.set_ylabel('Feature 2'); ax2.set_title('Classificação do Perceptron'); ax2.legend(); ax2.grid(True, alpha=0.3)
    return fig

def plot_neural_network(layers):
    fig, ax = plt.subplots(figsize=(12, 8)); layer_positions = np.linspace(1, 10, len(layers)); max_neurons = max(layers) if layers else 1; colors = ['lightblue', 'orange', 'lightgreen']
    for i, (pos, neurons) in enumerate(zip(layer_positions, layers)):
        y_positions = np.linspace(1, max_neurons, neurons); color = colors[i % len(colors)]
        for j, y_pos in enumerate(y_positions):
            ax.add_patch(Circle((pos, y_pos), 0.2, color=color, ec='black'))
            if i < len(layers) - 1:
                next_y_positions = np.linspace(1, max_neurons, layers[i + 1])
                for next_y in next_y_positions:
                    ax.plot([pos + 0.2, layer_positions[i + 1] - 0.2], [y_pos, next_y], 'k-', alpha=0.3, linewidth=0.5)
    labels = ['Entrada', 'Oculta', 'Saída'];
    for i, pos in enumerate(layer_positions):
        if i < len(labels): ax.text(pos, max_neurons + 0.5, labels[i], ha='center', fontweight='bold')
    ax.set_xlim(0, 11); ax.set_ylim(0, max_neurons + 1); ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Arquitetura da Rede Neural', fontsize=16, fontweight='bold')
    return fig

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

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5; y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
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
        st.markdown("""<div class="info-box">As <strong>Redes Neurais Artificiais (RNAs)</strong> são modelos computacionais inspirados pela estrutura e funcionamento do cérebro humano. Elas constituem o núcleo de muitos dos avanços em Inteligência Artificial e são capazes de aprender padrões complexos a partir de dados através de um processo de treinamento.<h4>Componentes Fundamentais:</h4><ul><li><b>Neurônios (ou Nós):</b> As unidades computacionais básicas que recebem entradas, processam-nas e geram uma saída.</li><li><b>Conexões e Pesos:</b> Cada conexão entre neurônios possui um peso associado, que modula a força do sinal. O aprendizado ocorre pelo ajuste desses pesos.</li><li><b>Bias:</b> Um parâmetro extra, similar ao intercepto em uma regressão linear, que permite deslocar a função de ativação.</li><li><b>Função de Ativação:</b> Determina a saída do neurônio, introduzindo não-linearidades que permitem à rede aprender padrões complexos.</li></ul></div>""", unsafe_allow_html=True)
    with col2:
        st.image("images.png", caption="Modelo de um neurônio artificial.", use_container_width=True)

elif secao == "🔧 Perceptron":
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
            st.pyplot(plot_perceptron(perceptron.weights,perceptron.bias,X_train,y_train,"Resultado Final"))

elif secao == "🌐 Redes Neurais":
    st.markdown('<h2 class="section-header">Redes Neurais Multicamadas (MLP)</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1]);
    with col1:
        st.markdown("""<div class="info-box">As <strong>Redes Neurais Multicamadas</strong> (MLPs - Multi-Layer Perceptrons) superam as limitações do perceptron simples, sendo capazes de resolver problemas não-linearmente separáveis através de camadas ocultas e funções de ativação não-lineares.</div>""", unsafe_allow_html=True)
        st.markdown("### 🏗️ Arquitetura\n- **Camada de Entrada**: Recebe os dados brutos.\n- **Camadas Ocultas**: Processam e extraem características dos dados.\n- **Camada de Saída**: Produz o resultado final.")
    with col2:
        st.subheader("🎮 Configurar Arquitetura"); input_neurons, hidden_neurons, output_neurons = st.number_input("Neurônios de Entrada",2,10,3,key='nn_in'), st.number_input("Neurônios Ocultos",2,20,5,key='nn_hid'), st.number_input("Neurônios de Saída",1,5,1,key='nn_out'); layers = [input_neurons, hidden_neurons, output_neurons]
    st.subheader("📊 Visualização da Arquitetura"); st.markdown("Use os controles acima para montar a arquitetura da sua rede."); st.pyplot(plot_neural_network(layers))
    st.markdown("---"); st.subheader("📈 Funções de Ativação Comuns"); st.markdown("Essas funções introduzem não-linearidade, permitindo que a rede aprenda padrões complexos."); x = np.linspace(-5, 5, 100); sigmoid = 1/(1+np.exp(-x)); relu = np.maximum(0,x); tanh = np.tanh(x)
    fig_activation, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,8)); fig_activation.suptitle("Visualização de Funções de Ativação",fontsize=16)
    ax1.plot(x,sigmoid,'b-',linewidth=2); ax1.set_title('Sigmoid'); ax1.grid(True,alpha=0.3); ax1.set_ylim(-.1,1.1)
    ax2.plot(x,relu,'r-',linewidth=2); ax2.set_title('ReLU'); ax2.grid(True,alpha=0.3)
    ax3.plot(x,tanh,'g-',linewidth=2); ax3.set_title('Tanh'); ax3.grid(True,alpha=0.3); ax3.set_ylim(-1.1,1.1)
    ax4.plot(x,sigmoid,'b-',label='Sigmoid'); ax4.plot(x,relu,'r-',label='ReLU'); ax4.plot(x,tanh,'g-',label='Tanh'); ax4.set_title('Comparação'); ax4.legend(); ax4.grid(True,alpha=0.3)
    plt.tight_layout(rect=[0,0.03,1,0.95]); st.pyplot(fig_activation)

elif secao == "🔄 Backpropagation":
    st.markdown('<h2 class="section-header">Backpropagation: Um Passo de Cada Vez</h2>', unsafe_allow_html=True)
    st.markdown("""O Backpropagation pode parecer uma "caixa preta". Vamos abri-la e executar um único passo de treinamento de forma interativa. Veremos exatamente como a rede usa o erro para descobrir como ajustar seus pesos.\n\nNosso cenário: uma rede com **1 neurônio**, **2 entradas**, e função de ativação **Sigmoid**.""")
    col1, col2 = st.columns([1,1.2])
    with col1:
        st.subheader("⚙️ Controles"); learning_rate = st.slider("Taxa de Aprendizado (η)",.01,1.,.5,.01)
        if 'bp_step' not in st.session_state: st.session_state.bp_step = 0
        def next_step(): st.session_state.bp_step += 1
        def reset_steps(): st.session_state.bp_step,st.session_state.w1,st.session_state.w2,st.session_state.bias=0,.3,-.5,.1
        if 'w1' not in st.session_state: reset_steps()
        c1,c2=st.columns(2);c1.button("Próximo Passo ➡️",on_click=next_step,type="primary",use_container_width=True);c2.button("Reiniciar 🔄",on_click=reset_steps,use_container_width=True)
        x=np.array([2.,3.]);y_real=1.;w=np.array([st.session_state.w1,st.session_state.w2]);bias=st.session_state.bias
    with col2:
        st.subheader("Visualização da Descida do Gradiente");w_space=np.linspace(-1,1,100);error_space=(w_space-.7)**2
        fig,ax=plt.subplots();ax.plot(w_space,error_space,label="Superfície de Erro");ax.set_xlabel("Valor do Peso (Ex: w1)");ax.set_ylabel("Erro");ax.set_title("O Objetivo: Atingir o Mínimo do Erro");ax.plot(st.session_state.w1,(st.session_state.w1-.7)**2,'ro',markersize=10,label="Peso Atual")
        if st.session_state.bp_step>=5:
            w_novo_calculado = w[0]-learning_rate*st.session_state.grad_w1
            ax.plot(w_novo_calculado,(w_novo_calculado-.7)**2,'go',markersize=10,label="Peso Novo")
            ax.annotate("",xy=(w_novo_calculado,(w_novo_calculado-.7)**2),xytext=(st.session_state.w1,(st.session_state.w1-.7)**2),arrowprops=dict(arrowstyle="->",color="purple",lw=2))
        ax.legend();st.pyplot(fig)
    st.markdown("---"); st.subheader("🔍 O Processo Detalhado"); st.markdown("##### 🏁 Estado Inicial"); st.markdown(f"- **Entradas:** `x₁ = {x[0]}`, `x₂ = {x[1]}`\n- **Alvo Real:** `y_real = {y_real}`\n- **Pesos Iniciais:** `w₁ = {w[0]:.2f}`, `w₂ = {w[1]:.2f}`\n- **Bias Inicial:** `b = {bias:.2f}`")
    if st.session_state.bp_step>=1:
        with st.container(border=True):
            st.markdown("##### Passo 1: Forward Pass - Calcular a Predição da Rede");z=np.dot(w,x)+bias;predicao=1/(1+np.exp(-z))
            st.latex(r"z=(w₁\times x₁)+(w₂\times x₂)+b");st.markdown(f"`z=({w[0]:.2f}×{x[0]})+({w[1]:.2f}×{x[1]})+{bias:.2f}={z:.4f}`")
            st.latex(r"\text{predição (a)}=\sigma(z)=\frac{1}{1+e^{-z}}");st.markdown(f"`predição = σ({z:.4f})={predicao:.4f}`")
            st.session_state.predicao,st.session_state.z=predicao,z
    if st.session_state.bp_step>=2:
        with st.container(border=True):
            st.markdown("##### Passo 2: Calcular o Erro da Predição");erro=.5*(y_real-st.session_state.predicao)**2
            st.latex(r"E=\frac{1}{2}(y_{real}-\text{predição})^2");st.markdown(f"`E=0.5*({y_real}-{st.session_state.predicao:.4f})²={erro:.4f}`")
    if st.session_state.bp_step>=3:
        with st.container(border=True):
            st.markdown("##### Passo 3: Backward Pass - Calcular os Gradientes (A Mágica da Regra da Cadeia)");st.info("O objetivo é descobrir a contribuição de cada peso para o erro. Usamos a Regra da Cadeia do cálculo: `∂E/∂w=(∂E/∂predição)*(∂predição/∂z)*(∂z/∂w)`")
            dE_dpred,dpred_dz=st.session_state.predicao-y_real,st.session_state.predicao*(1-st.session_state.predicao)
            st.markdown("**Componentes do Gradiente:**");st.markdown(f"- `∂E/∂predição=predição-y_real={dE_dpred:.4f}`");st.markdown(f"- `∂predição/∂z=σ(z)*(1-σ(z))={dpred_dz:.4f}`");st.markdown(f"- `∂z/∂w₁=x₁={x[0]}`")
            st.session_state.dE_dpred,st.session_state.dpred_dz=dE_dpred,dpred_dz
    if st.session_state.bp_step>=4:
        with st.container(border=True):
            st.markdown("##### Passo 4: Gradientes Finais");grad_w1,grad_w2,grad_bias=st.session_state.dE_dpred*st.session_state.dpred_dz*x[0],st.session_state.dE_dpred*st.session_state.dpred_dz*x[1],st.session_state.dE_dpred*st.session_state.dpred_dz*1.
            st.latex(r"\frac{\partial E}{\partial w_1}=(%.2f)\times(%.2f)\times(%.1f)=%.4f"%(dE_dpred,st.session_state.dpred_dz,x[0],grad_w1))
            st.latex(r"\frac{\partial E}{\partial w_2}=(%.2f)\times(%.2f)\times(%.1f)=%.4f"%(dE_dpred,st.session_state.dpred_dz,x[1],grad_w2))
            st.latex(r"\frac{\partial E}{\partial b}=(%.2f)\times(%.2f)\times(1)=%.4f"%(dE_dpred,st.session_state.dpred_dz,grad_bias))
            st.session_state.grad_w1,st.session_state.grad_w2,st.session_state.grad_bias=grad_w1,grad_w2,grad_bias
    if st.session_state.bp_step>=5:
        with st.container(border=True):
            st.markdown("##### Passo 5: Atualizar os Pesos - O Aprendizado Acontece!");st.success("Finalmente, ajustamos os pesos na direção OPOSTA ao gradiente, multiplicando pela taxa de aprendizado.")
            w_novo1,w_novo2,bias_novo=w[0]-learning_rate*st.session_state.grad_w1,w[1]-learning_rate*st.session_state.grad_w2,bias-learning_rate*st.session_state.grad_bias
            st.latex(r"w_{novo}=w_{antigo}-\eta\times\frac{\partial E}{\partial w}");st.markdown("**Novos Pesos:**");st.markdown(f"- `w₁_novo={w[0]:.2f}-{learning_rate}×({st.session_state.grad_w1:.4f})={w_novo1:.4f}`\n- `w₂_novo={w[1]:.2f}-{learning_rate}×({st.session_state.grad_w2:.4f})={w_novo2:.4f}`\n- `b_novo={bias:.2f}-{learning_rate}×({st.session_state.grad_bias:.4f})={bias_novo:.4f}`")
            st.markdown("\nEstes seriam os novos pesos para o próximo ciclo de treinamento!")
    if st.session_state.bp_step > 5: st.balloons();st.info("Você completou um passo de backpropagation! Clique em 'Reiniciar' para ver o processo novamente com diferentes taxas de aprendizado.")

elif secao == "🧠 MLP em Ação":
    st.markdown('<h2 class="section-header">MLP em Ação: Resolvendo Problemas Não-Lineares</h2>', unsafe_allow_html=True)
    st.markdown("""Vamos ver o MLP resolver um problema que o Perceptron não consegue. Para avaliar o modelo de forma realista, vamos dividir nossos dados em um conjunto de **treino** (para ensinar o modelo) e um conjunto de **teste** (para avaliar seu poder de generalização com dados nunca vistos).""")
    col1, col2 = st.columns([1,2]);
    with col1:
        st.subheader("⚙️ Parâmetros do Modelo");hidden_layers=st.slider("Neurônios na Camada Oculta",2,64,10,key='mlp_h');activation=st.selectbox("Função de Ativação",["relu","tanh","logistic"],key='mlp_a');learning_rate_mlp=st.slider("Taxa de Aprendizado",.001,.1,.01,format="%.3f",key='mlp_lr');epochs_mlp=st.slider("Máximo de Épocas",100,1000,300,key='mlp_e');train_button=st.button("🚀 Treinar e Avaliar MLP",type="primary")
    X,y=make_circles(n_samples=300,noise=.15,factor=.5,random_state=1);X_scaled=StandardScaler().fit_transform(X);X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=.3,random_state=42)
    with col2:
        st.subheader("📊 Dados e Resultado")
        if train_button:
            model = MLPClassifier(hidden_layer_sizes=(hidden_layers,),activation=activation,solver='adam',learning_rate_init=learning_rate_mlp,max_iter=epochs_mlp,random_state=1,early_stopping=True)
            model.fit(X_train,y_train);st.session_state.mlp_model=model;acc_train,acc_test=model.score(X_train,y_train),model.score(X_test,y_test);st.session_state.mlp_scores=(acc_train,acc_test);st.success("✅ Modelo treinado e avaliado!")
        if 'mlp_model' in st.session_state:
            acc_train,acc_test=st.session_state.mlp_scores;st.metric("Acurácia no Treino (Dados Vistos)",f"{acc_train:.2%}");st.metric("Acurácia no Teste (Dados Novos - Generalização)",f"{acc_test:.2%}")
            st.plotly_chart(plot_decision_boundary(st.session_state.mlp_model,X_scaled,y),use_container_width=True);st.info("Observe como a acurácia no teste é geralmente um pouco menor que no treino. Esta é a medida mais honesta do desempenho do modelo!")
        else:
            st.plotly_chart(px.scatter(x=X_scaled[:,0],y=X_scaled[:,1],color=y.astype(str),title="Dataset 'Círculos' (Não-Linearmente Separável)"),use_container_width=True)

elif secao == "🖼️ CNNs":
    st.markdown('<h2 class="section-header">Redes Neurais Convolucionais (CNNs)</h2>', unsafe_allow_html=True)
    st.markdown("CNNs são uma classe de redes neurais especializada no processamento de dados com uma topologia de grade, como imagens.");
    with st.expander("👁️ A Operação de Convolução",expanded=True):
        st.markdown("A convolução aplica um **filtro (ou kernel)** sobre a imagem, criando um **mapa de características** que destaca padrões como bordas ou texturas.")
        col1,col2 = st.columns([1,2]);
        with col1:
            st.subheader("Filtros Interativos");image_data=np.zeros((10,10));image_data[2:8,2:8]=10;kernels={"Detector de Borda Vertical":np.array([[1,0,-1],[2,0,-2],[1,0,-1]]),"Detector de Borda Horizontal":np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),"Sharpen":np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])};kernel_choice=st.selectbox("Escolha um Kernel:",list(kernels.keys()));kernel=kernels[kernel_choice]
        with col2:
            convolved_image = convolve2d(image_data,kernel,mode='valid');fig,ax = plt.subplots(1,3,figsize=(12,4));ax[0].imshow(image_data,cmap='gray');ax[0].set_title('Imagem de Entrada');ax[1].imshow(kernel,cmap='gray');ax[1].set_title('Kernel');ax[2].imshow(convolved_image,cmap='gray');ax[2].set_title('Mapa de Características');
            for a in ax: a.axis('off');st.pyplot(fig)

elif secao == "📜 RNNs":
    st.markdown('<h2 class="section-header">Redes Neurais Recorrentes (RNNs)</h2>', unsafe_allow_html=True)
    st.markdown("RNNs são projetadas para trabalhar com dados sequenciais. Sua característica definidora é a **conexão recorrente**, que cria uma **memória** para reter informações sobre o passado.")
    st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png",caption="Uma RNN 'desdobrada' no tempo. Fonte: Chris Olah's Blog.")
    with st.expander("🧠 Solução para Memória de Longo Prazo: LSTM e GRU",expanded=True):
        st.markdown("**Long Short-Term Memory (LSTM)** e **Gated Recurrent Unit (GRU)** usam 'portões' (gates) para regular o fluxo de informação, permitindo que a rede aprenda a reter ou descartar informações de forma seletiva.")
        st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png",caption="A estrutura de uma célula LSTM com seus portões. Fonte: Chris Olah's Blog.")

elif secao == "🤖 Transformers":
    st.markdown('<h2 class="section-header">🤖 Transformers com Vocabulário em Português (BERT)</h2>', unsafe_allow_html=True)

    if not TORCH_AVAILABLE:
        st.error("Para executar esta seção, por favor instale: `pip install torch seaborn transformers`")
    else:
        modelo_hf = "neuralmind/bert-base-portuguese-cased"
        with st.spinner(f"Baixando tokenizer '{modelo_hf}'... (Acontece apenas na primeira vez)"):
            tokenizer = carregar_tokenizer_hf(modelo_hf)

        if tokenizer:
            @st.cache_resource
            def get_demo_model(_tokenizer):
                return DemonstradorGPT(tokenizer=_tokenizer)
            
            demo_gpt = get_demo_model(tokenizer)

            st.markdown(f"""
            <div class="info-box">
            Estamos usando um tokenizador profissional, treinado para o português: <strong>{modelo_hf}</strong>.
            Ele possui um vocabulário de <strong>{tokenizer.vocab_size:,}</strong> tokens, incluindo subpalavras, o que o torna extremamente robusto.
            <br><br>
            Note como palavras complexas podem ser quebradas em pedaços menores (ex: "aprendizagem" -> "aprend", "##izagem"). Esta é uma técnica chave dos modelos modernos.
            </div>
            """, unsafe_allow_html=True)

            st.subheader("🧪 Demonstração Interativa")
            texto_usuario = st.text_input("Digite uma frase em português para análise:", "O gato subiu")

            if st.button("Analisar e Gerar Texto (Antes do Treino)", type="primary", use_container_width=True):
                if texto_usuario.strip():
                    st.session_state.texto_analisado = texto_usuario; st.session_state.analysis_done = True
                    st.session_state.training_done = False; st.session_state.texto_depois_treino = ""
                else:
                    st.warning("Por favor, digite uma frase.")

            if st.session_state.get('analysis_done', False):
                texto = st.session_state.texto_analisado
                tokens_ids = tokenizer.encode(texto)
                tokens_str = tokenizer.convert_ids_to_tokens(tokens_ids)
                
                st.markdown("---"); st.subheader("Análise do Modelo NÃO TREINADO")

                with st.expander("Passo 1: Tokenização Profissional (BERT)", expanded=True):
                    df_tokens = pd.DataFrame({"Token": tokens_str, "ID do Token": tokens_ids})
                    st.table(df_tokens.set_index("Token"))
                    if tokenizer.unk_token_id in tokens_ids:
                        st.warning(f"Atenção: O token `{tokenizer.unk_token}` apareceu! A palavra original era desconhecida.")
                    else:
                        st.success("Todas as palavras/subpalavras foram reconhecidas pelo vocabulário do BERT.")

                st.markdown("---"); st.subheader("⚡ O Efeito do Treinamento: Antes vs. Depois")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔴 Antes do Treino")
                    st.warning("Com pesos aleatórios, as previsões são sem sentido. As probabilidades são tão baixas que arredondam para 0%.")
                    
                    # Adicionando o controle de Temperatura
                    st.info("💡 **Dica:** Usamos uma 'Temperatura' baixa para forçar o modelo a ser mais confiante e tornar as probabilidades visíveis. Sem isso, todas seriam 0.0%.")
                    temperature = st.slider("Temperatura de Geração", 0.1, 2.0, 0.7, 0.1, key="temp_antes")

                    demo_gpt.reset_model()
                    prompt_ids = tokenizer.encode(texto, return_tensors='pt')
                    
                    with torch.no_grad():
                        for i in range(3):
                            st.markdown(f"**Prevendo a palavra {i+1}**")
                            current_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
                            st.markdown(f"**Contexto:** `{current_text}`")
                            
                            logits, _ = demo_gpt.model(prompt_ids)
                            # APLICAÇÃO DA TEMPERATURA
                            next_token_logits = logits[0, -1, :] / temperature
                            probs = F.softmax(next_token_logits, dim=-1)
                            top_probs, top_indices = torch.topk(probs, 3)
                            
                            pred_cols = st.columns(3)
                            for j, (p, idx) in enumerate(zip(top_probs, top_indices)):
                                token_name = tokenizer.decode([idx.item()])
                                pred_cols[j].metric(label=f"'{token_name}'", value=f"{p.item():.2%}") # Aumentei a precisão
                            
                            next_token_id = top_indices[0].view(1, 1)
                            prompt_ids = torch.cat([prompt_ids, next_token_id], dim=1)

                    st.markdown("**Resultado Final (Não Treinado):**")
                    st.code(tokenizer.decode(prompt_ids[0], skip_special_tokens=True), language="text")

                with col2:
                    st.markdown("#### 🟢 Depois do Treino")
                    if not st.session_state.get('training_done', False):
                        if st.button("Simular Treinamento Agora"):
                            with st.spinner("Ajustando os pesos do modelo..."):
                                log_placeholder = st.empty()
                                demo_gpt.simular_treinamento(log_placeholder)
                            st.session_state.texto_depois_treino = demo_gpt.gerar_texto(texto, max_tokens=5)
                            st.session_state.training_done = True
                            st.rerun()
                    
                    if st.session_state.get('training_done', False):
                        st.success("Após aprender com exemplos, o modelo agora faz predições mais lógicas.")
                        st.markdown(f"**Prompt:** `{texto}`"); st.markdown("**Resultado:**")
                        st.code(st.session_state.texto_depois_treino, language="text")
elif secao == "🎮 Playground Interativo":
    st.markdown('<h2 class="section-header">Playground Interativo de MLP</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.subheader("🎯 Dados"); dataset_type = st.selectbox("Tipo de Dataset", ["Círculos", "Luas", "Linearmente Separável"]); n_samples = st.slider("Amostras", 100, 1000, 300)
    with col2:
        st.subheader("🏗️ Arquitetura"); hl_1 = st.slider("Neurônios Camada 1", 1, 50, 10); hl_2 = st.slider("Neurônios Camada 2", 0, 50, 5); activation_pg = st.selectbox("Ativação", ["relu", "tanh"], key='pg_act')
    if dataset_type == "Linearmente Separável": X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
    elif dataset_type == "Círculos": X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    else: X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    with col3:
        st.subheader("📊 Dados de Entrada"); fig_data = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], color=y.astype(str)); fig_data.update_layout(showlegend=False, title=f"Dataset: {dataset_type}"); st.plotly_chart(fig_data, use_container_width=True)
    if st.button("🚀 TREINAR REDE NEURAL NO PLAYGROUND", type="primary"):
        hidden_layers_pg = (hl_1,) if hl_2 == 0 else (hl_1, hl_2)
        model_pg = MLPClassifier(hidden_layer_sizes=hidden_layers_pg, activation=activation_pg, solver='adam', max_iter=500, random_state=1, early_stopping=True, n_iter_no_change=20)
        with st.spinner("Treinando o modelo..."): model_pg.fit(X_scaled, y)
        st.session_state.pg_model = model_pg; st.success("✅ Treinamento Concluído!")
    if 'pg_model' in st.session_state:
        res_col1, res_col2 = st.columns(2); model = st.session_state.pg_model
        with res_col1:
            st.markdown("#### Fronteira de Decisão"); boundary_fig = plot_decision_boundary(model, X_scaled, y); st.plotly_chart(boundary_fig, use_container_width=True)
        with res_col2:
            st.markdown("#### Métricas"); st.metric("Acurácia no Treino", f"{model.score(X_scaled, y):.2%}"); st.metric("Iterações", f"{model.n_iter_}"); st.metric("Perda Final", f"{model.loss_:.4f}")

# --- RODAPÉ ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><strong>Guia Didático de Redes Neurais</strong> | Desenvolvido para o aprendizado aprofundado de conceitos de Machine Learning.</div>", unsafe_allow_html=True)