import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def mostrar():
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