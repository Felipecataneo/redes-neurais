# secao_transformers.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List, Tuple, Dict
import random

# --- DEPENDÊNCIAS DO TRANSFORMER ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- FUNÇÃO PARA CARREGAR TOKENIZER ---
@st.cache_resource
def carregar_tokenizer_hf(modelo_nome: str):
    """Baixa e carrega um tokenizer da Hugging Face, com cache."""
    try:
        return AutoTokenizer.from_pretrained(modelo_nome)
    except Exception as e:
        st.error(f"Falha ao carregar o tokenizer '{modelo_nome}': {e}")
        return None

# --- CLASSES E LÓGICA DA SEÇÃO TRANSFORMERS ---
if TORCH_AVAILABLE:
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, num_heads: int):
            super().__init__()
            self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size, seq_len, _ = x.shape
            Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.W_o(attn_output), attn_weights

        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)
            return output, attn_weights

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.attention = MultiHeadAttention(d_model, num_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, mask=None):
            attn_output, attn_weights = self.attention(x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_output))
            return x, attn_weights

    class GPTMelhorado(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_model * 4, dropout=0.1) for _ in range(num_layers)
            ])
            self.ln_final = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None: torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        def forward(self, input_ids: torch.Tensor):
            seq_len = input_ids.shape[1]
            device = input_ids.device
            token_embeds = self.token_embedding(input_ids)
            pos_embeds = self.position_embedding[:seq_len]
            x = token_embeds + pos_embeds
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
            attention_weights_list = []
            for block in self.transformer_blocks:
                x, attn_weights = block(x, mask)
                attention_weights_list.append(attn_weights)
            x = self.ln_final(x)
            logits = self.lm_head(x)
            return logits, attention_weights_list

    class DemonstradorGPTMelhorado:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.model = GPTMelhorado(
                vocab_size=tokenizer.vocab_size, d_model=128, num_heads=4, num_layers=3, max_seq_len=128
            )
            self.initial_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.dataset_treino = [
                "o gato subiu no telhado e desceu pela chaminé", "inteligência artificial vai revolucionar o mundo dos negócios",
                "o brasil é um país tropical com muitas belezas naturais", "programação em python é muito útil para ciência de dados",
                "aprender sobre transformers ajuda a entender como funcionam os modelos de linguagem", "são paulo é a maior cidade do brasil em população",
                "machine learning e deep learning são áreas em crescimento", "processamento de linguagem natural permite que computadores entendam texto",
                "redes neurais profundas conseguem aprender padrões complexos nos dados", "tokenização é o primeiro passo para processar texto em modelos de IA"
            ]

        def reset_model(self):
            self.model.load_state_dict(self.initial_state_dict)

        def preprocessar_texto(self, texto: str, max_length: int = 64) -> torch.Tensor:
            # ✅ CORREÇÃO: Não adicionar tokens especiais ([CLS], [SEP]) ao prompt.
            return self.tokenizer.encode(
                texto, max_length=max_length, truncation=True, add_special_tokens=False, return_tensors='pt'
            )

        def gerar_texto(self, prompt: str, max_tokens: int = 10, temperature: float = 0.8, top_k: int = 50) -> Tuple[str, List[Dict]]:
            self.model.eval()
            input_ids = self.preprocessar_texto(prompt)
            geracoes_info = []
            with torch.no_grad():
                for step in range(max_tokens):
                    if input_ids.shape[1] >= self.model.position_embedding.shape[0]: break
                    outputs = self.model(input_ids)
                    logits = outputs[0]
                    next_token_logits = logits[0, -1, :] / temperature
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                        next_token_logits[top_k_indices] = top_k_logits
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    top_probs, top_indices = torch.topk(probs, 5)
                    geracoes_info.append({
                        'step': step, 'contexto': self.tokenizer.decode(input_ids[0], skip_special_tokens=True),
                        'top_tokens': [(self.tokenizer.decode([idx.item()], skip_special_tokens=True), prob.item()) for idx, prob in zip(top_indices, top_probs)],
                        'escolhido': self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)
                    })
                    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            texto_final = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            return texto_final, geracoes_info

        def simular_treinamento_melhorado(self, st_log_placeholder, epochs: int = 30):
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            log_text = f"🏋️ Iniciando treinamento avançado...\n📊 Dataset: {len(self.dataset_treino)} frases\n🎯 Parâmetros: {sum(p.numel() for p in self.model.parameters()):,}\n\n"
            st_log_placeholder.markdown(f"```\n{log_text}\n```")
            perdas_historico = []
            for epoca in range(epochs):
                total_loss, num_batches = 0, 0
                random.shuffle(self.dataset_treino)
                for frase in self.dataset_treino:
                    # ✅ CORREÇÃO: Não adicionar tokens especiais ([CLS], [SEP]) aos dados de treino.
                    tokens = self.tokenizer.encode(frase, max_length=64, truncation=True, add_special_tokens=False)
                    if len(tokens) < 3: continue
                    input_ids = torch.tensor([tokens[:-1]]); targets = torch.tensor([tokens[1:]])
                    optimizer.zero_grad()
                    logits, _ = self.model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item(); num_batches += 1
                scheduler.step()
                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                    perdas_historico.append(avg_loss)
                    if (epoca + 1) % 5 == 0 or epoca < 5:
                        log_text += f"  Época {epoca+1:2d}/{epochs} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}\n"
                        st_log_placeholder.markdown(f"```\n{log_text}\n```")
                        time.sleep(0.1)
            log_text += f"\n✅ Treinamento concluído!\n📉 Loss final: {perdas_historico[-1]:.4f}\n"
            if len(perdas_historico) > 1 and perdas_historico[0] > 0:
                 log_text += f"📈 Melhoria de {((perdas_historico[0] - perdas_historico[-1]) / perdas_historico[0] * 100):.1f}%"
            st_log_placeholder.markdown(f"```\n{log_text}\n```")
            return perdas_historico

        def visualizar_atencao_melhorada(self, texto: str) -> plt.Figure:
            self.model.eval()
            input_ids = self.preprocessar_texto(texto)
            if input_ids.shape[1] == 0: return plt.figure()
            tokens_str = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            with torch.no_grad():
                _, attention_weights_list = self.model(input_ids)
            attn_weights = attention_weights_list[0][0, 0].cpu().numpy()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            sns.heatmap(attn_weights, xticklabels=tokens_str, yticklabels=tokens_str, annot=True, fmt='.2f', cmap='Blues', ax=ax1, cbar_kws={'label': 'Peso de Atenção'})
            ax1.set_title('Matriz de Atenção\n(1ª Camada, 1ª Cabeça)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Tokens (Keys - Fonte da Informação)'); ax1.set_ylabel('Tokens (Queries - Buscando Informação)')
            attn_mean = np.mean(attn_weights, axis=0)
            bars = ax2.bar(range(len(tokens_str)), attn_mean, color='skyblue', alpha=0.7)
            ax2.set_xlabel('Posição do Token'); ax2.set_ylabel('Atenção Média Recebida')
            ax2.set_title('Importância dos Tokens\n(Média de Atenção)', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(tokens_str))); ax2.set_xticklabels(tokens_str, rotation=45, ha='right')
            if attn_mean.size > 0:
                max_idx = np.argmax(attn_mean)
                bars[max_idx].set_color('orange'); bars[max_idx].set_alpha(1.0)
            plt.tight_layout()
            return fig

def mostrar():
    # O restante da função 'mostrar' permanece o mesmo, pois sua UI já estava excelente.
    # Colei-a aqui para garantir que você tenha o arquivo completo e funcional.
    st.markdown('<h2 class="section-header">🤖 Transformers Avançados - Português com BERT</h2>', unsafe_allow_html=True)

    if not TORCH_AVAILABLE:
        st.error("⚠️ Dependências não encontradas. Para executar esta seção, instale:\n```bash\npip install torch seaborn transformers\n```")
        return

    modelo_hf = "neuralmind/bert-base-portuguese-cased"
    
    with st.spinner(f"🔄 Carregando tokenizer '{modelo_hf}'... (Primeira vez pode demorar)"):
        tokenizer = carregar_tokenizer_hf(modelo_hf)

    if not tokenizer: return

    @st.cache_resource
    def get_demo_model(_tokenizer):
        return DemonstradorGPTMelhorado(tokenizer=_tokenizer)
    
    demo_gpt = get_demo_model(tokenizer)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
    <h3>🧠 Sobre o Modelo</h3>
    <ul>
    <li><strong>Tokenizer:</strong> {modelo_hf}</li>
    <li><strong>Vocabulário:</strong> {tokenizer.vocab_size:,} tokens</li>
    <li><strong>Parâmetros:</strong> {sum(p.numel() for p in demo_gpt.model.parameters()):,}</li>
    <li><strong>Arquitetura:</strong> {len(demo_gpt.model.transformer_blocks)} camadas, {demo_gpt.model.transformer_blocks[0].attention.num_heads} cabeças, {demo_gpt.model.token_embedding.embedding_dim} dimensões</li>
    </ul>
    <p>💡 <em>Este tokenizer foi treinado especificamente para português e usa subpalavras (BPE) para lidar com palavras complexas e variações morfológicas.</em></p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎯 Demonstração Interativa")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        texto_usuario = st.text_input("Digite uma frase em português para análise:", "O aprendizado de máquina está revolucionando", help="Experimente frases sobre tecnologia, ciência ou temas do cotidiano")
    with col2:
        temperatura = st.slider("Temperatura", 0.1, 2.0, 0.8, 0.1, help="Controla a criatividade: baixa = mais conservador, alta = mais criativo")

    if st.button("🚀 Analisar e Demonstrar", type="primary", use_container_width=True):
        if texto_usuario.strip():
            st.session_state.texto_analisado = texto_usuario
            st.session_state.temperatura = temperatura
            st.session_state.analysis_done = True
            st.session_state.training_done = False
            st.rerun()
        else:
            st.warning("Por favor, digite uma frase.")

    if st.session_state.get('analysis_done', False):
        texto = st.session_state.texto_analisado
        temp = st.session_state.get('temperatura', 0.8)
        
        st.markdown("---")
        st.subheader("🔍 Análise de Tokenização")
        tokens_ids_com_especiais = tokenizer.encode(texto, max_length=64, truncation=True)
        tokens_str_com_especiais = tokenizer.convert_ids_to_tokens(tokens_ids_com_especiais)
        
        with st.expander("📝 Detalhes da Tokenização (Como o BERT Vê)", expanded=True):
            token_html = ""
            for token in tokens_str_com_especiais:
                color = "lightcoral" if token in ['[CLS]', '[SEP]'] else ("lightblue" if token.startswith("##") else "lightgreen")
                token_html += f'<span style="background-color: {color}; padding: 2px 4px; margin: 2px; border-radius: 3px; font-family: monospace;">{token}</span>'
            st.markdown(f"**Tokens:** {token_html}", unsafe_allow_html=True)
            st.info("Para a nossa tarefa de geração, removemos os tokens especiais `[CLS]` e `[SEP]` antes de alimentar o modelo.")

        st.markdown("---")
        st.subheader("⚡ O Poder do Treinamento: Transformação Completa")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔴 ANTES: Modelo Não Treinado")
            st.info("🎲 **Previsões aleatórias** - O modelo ainda não aprendeu padrões da linguagem")
            demo_gpt.reset_model()
            with st.spinner("Gerando com modelo não treinado..."):
                texto_antes, info_antes = demo_gpt.gerar_texto(texto, max_tokens=8, temperature=temp)
            st.markdown("**Resultado:**"); st.code(texto_antes, language="text")
            if info_antes:
                st.markdown("**Primeiras 3 previsões:**")
                for i, info in enumerate(info_antes[:3]):
                    with st.expander(f"Passo {i+1}: '{info['escolhido']}'"):
                        st.markdown(f"**Contexto:** `{info['contexto']}`")
                        cols = st.columns(len(info['top_tokens']))
                        for j, (token, prob) in enumerate(info['top_tokens']):
                            cols[j].metric(f"'{token}'", f"{prob:.1%}")
        
        with col2:
            st.markdown("#### 🟢 DEPOIS: Modelo Treinado")
            if not st.session_state.get('training_done', False):
                if st.button("🏋️ Iniciar Treinamento Avançado", use_container_width=True):
                    with st.status("Treinando modelo...", expanded=True) as status:
                        log_placeholder = st.empty()
                        perdas = demo_gpt.simular_treinamento_melhorado(log_placeholder)
                        texto_depois, info_depois = demo_gpt.gerar_texto(texto, max_tokens=8, temperature=temp)
                        st.session_state.update({
                            'texto_depois_treino': texto_depois, 'info_depois_treino': info_depois,
                            'perdas_treino': perdas, 'training_done': True, 'texto_antes_treino': texto_antes
                        })
                        status.update(label="✅ Treinamento concluído!", state="complete")
                    st.rerun()
            
            if st.session_state.get('training_done', False):
                st.success("🎯 **Previsões inteligentes** - O modelo aprendeu padrões linguísticos!")
                st.markdown("**Resultado:**"); st.code(st.session_state.texto_depois_treino, language="text")
                if st.session_state.get('info_depois_treino'):
                    st.markdown("**Primeiras 3 previsões (pós-treino):**")
                    for i, info in enumerate(st.session_state.info_depois_treino[:3]):
                        with st.expander(f"Passo {i+1}: '{info['escolhido']}'"):
                            st.markdown(f"**Contexto:** `{info['contexto']}`")
                            cols = st.columns(len(info['top_tokens']))
                            for j, (token, prob) in enumerate(info['top_tokens']):
                                cols[j].metric(f"'{token}'", f"{prob:.1%}")

        if st.session_state.get('training_done', False):
            st.markdown("---")
            st.subheader("📊 Análises Avançadas")
            
            tab1, tab2, tab3 = st.tabs(["📈 Curva de Aprendizado", "🎯 Matriz de Atenção", "🔬 Análise Detalhada"])
            
            with tab1:
                if st.session_state.get('perdas_treino'):
                    fig, ax = plt.subplots(figsize=(10, 6)); perdas = st.session_state.perdas_treino
                    ax.plot(perdas, 'b-', linewidth=2, label='Loss de Treinamento'); ax.set_xlabel('Época'); ax.set_ylabel('Loss (Cross-Entropy)')
                    ax.set_title('Evolução do Aprendizado', fontsize=14, fontweight='bold'); ax.grid(True, alpha=0.3); ax.legend()
                    st.pyplot(fig)
            
            with tab2:
                st.markdown("**Visualização de como o modelo 'presta atenção' nas palavras:**")
                fig_attn = demo_gpt.visualizar_atencao_melhorada(texto)
                st.pyplot(fig_attn)
                st.info("💡 **Interpretação:** Cores mais escuras indicam maior atenção.")
            
            with tab3:
                st.markdown("**Comparação Final:**")
                if st.session_state.get('texto_depois_treino'):
                    df_comparacao = pd.DataFrame({
                        'Aspecto': ['Texto Gerado', 'Qualidade', 'Coerência'],
                        'Antes do Treino': [st.session_state.texto_antes_treino[:50] + "...", "❌ Muito baixa", "❌ Sem sentido"],
                        'Depois do Treino': [st.session_state.texto_depois_treino[:50] + "...", "✅ Melhorada", "✅ Mais coerente"]
                    })
                    st.dataframe(df_comparacao, use_container_width=True)

    with st.expander("📚 Como Funcionam os Transformers?", expanded=False):
        st.markdown("""
        ### 🧠 Arquitetura Transformer: Uma Revolução em IA
        1. **Self-Attention:** Permite ao modelo pesar a importância de diferentes palavras na entrada.
        2. **Multi-Head Attention:** Executa a atenção em paralelo, permitindo focar em diferentes aspectos.
        3. **Processamento Paralelo:** Processa todas as palavras ao mesmo tempo, ao contrário de RNNs.
        4. **Positional Encoding:** Adiciona informação sobre a posição das palavras, já que o modelo não tem noção de sequência por si só.
        """)
    
    with st.expander("🔗 Recursos para Aprender Mais", expanded=False):
        st.markdown("""
        - **"Attention Is All You Need"**: Artigo original dos Transformers.
        - **Hugging Face Course**: Curso gratuito e prático.
        - **"The Illustrated Transformer"**: Post de blog visualmente incrível por Jay Alammar.
        """)

    if st.button("🔄 Reiniciar Demo", type="secondary"):
        keys_to_clear = ['analysis_done', 'training_done', 'texto_analisado', 'temperatura', 'texto_depois_treino', 'info_depois_treino', 'perdas_treino', 'texto_antes_treino']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()