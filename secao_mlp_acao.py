import streamlit as st
import plotly.express as px
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from visualizacoes import plot_decision_boundary

def mostrar():
    st.markdown('<h2 class="section-header">MLP em Ação: Resolvendo Problemas Não-Lineares</h2>', unsafe_allow_html=True)
    st.markdown("""Vamos ver o MLP resolver um problema que o Perceptron não consegue. Para avaliar o modelo de forma realista, vamos dividir nossos dados em um conjunto de **treino** (para ensinar o modelo) e um conjunto de **teste** (para avaliar seu poder de generalização com dados nunca vistos).""")
    col1, col2 = st.columns([1,2]);
    
    with col1:
        st.subheader("⚙️ Parâmetros do Modelo");
        hidden_layers=st.slider("Neurônios na Camada Oculta",2,64,10,key='mlp_h');
        activation=st.selectbox("Função de Ativação",["relu","tanh","logistic"],key='mlp_a');
        learning_rate_mlp=st.slider("Taxa de Aprendizado",.001,.1,.01,format="%.3f",key='mlp_lr');
        epochs_mlp=st.slider("Máximo de Épocas",100,1000,300,key='mlp_e');
        train_button=st.button("🚀 Treinar e Avaliar MLP",type="primary")
    
    X,y=make_circles(n_samples=300,noise=.15,factor=.5,random_state=1);
    X_scaled=StandardScaler().fit_transform(X);
    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=.3,random_state=42)
    
    with col2:
        st.subheader("📊 Dados e Resultado")
        if train_button:
            model = MLPClassifier(hidden_layer_sizes=(hidden_layers,),activation=activation,solver='adam',learning_rate_init=learning_rate_mlp,max_iter=epochs_mlp,random_state=1,early_stopping=True)
            model.fit(X_train,y_train);
            st.session_state.mlp_model=model;
            acc_train,acc_test=model.score(X_train,y_train),model.score(X_test,y_test);
            st.session_state.mlp_scores=(acc_train,acc_test);
            st.success("✅ Modelo treinado e avaliado!")
        
        if 'mlp_model' in st.session_state:
            acc_train,acc_test=st.session_state.mlp_scores;
            st.metric("Acurácia no Treino (Dados Vistos)",f"{acc_train:.2%}");
            st.metric("Acurácia no Teste (Dados Novos - Generalização)",f"{acc_test:.2%}")
            st.plotly_chart(plot_decision_boundary(st.session_state.mlp_model,X_scaled,y),use_container_width=True);
            st.info("Observe como a acurácia no teste é geralmente um pouco menor que no treino. Esta é a medida mais honesta do desempenho do modelo!")
        else:
            st.plotly_chart(px.scatter(x=X_scaled[:,0],y=X_scaled[:,1],color=y.astype(str),title="Dataset 'Círculos' (Não-Linearmente Separável)"),use_container_width=True)