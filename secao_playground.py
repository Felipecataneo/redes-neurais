import streamlit as st
import plotly.express as px
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from visualizacoes import plot_decision_boundary

def mostrar():
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