# 🧠 Guia Didático Interativo de Redes Neurais

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Streamlit-1.35+-ff4b4b.svg" alt="Streamlit">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT"></a>
</p>

<p align="center">
  <i>Uma plataforma interativa para desmistificar conceitos de Machine Learning, desde o Perceptron até a arquitetura Transformer, com foco em visualização e experimentação prática.</i>
</p>

---


## ✨ Sobre o Projeto

Este aplicativo web, desenvolvido com Streamlit, é um guia didático avançado sobre Redes Neurais. Foi projetado para estudantes, pesquisadores e entusiastas que desejam aprofundar seus conhecimentos não apenas na teoria, mas também na prática, visualizando o impacto de cada componente e hiperparâmetro em tempo real.

O projeto foi recentemente **refatorado para uma arquitetura modular**, tornando o código mais limpo, manutenível e fácil para novas contribuições.

## 📚 Seções e Recursos

O guia é dividido em módulos que constroem o conhecimento de forma progressiva:

| Seção                        | Recurso Principal                                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **🔍 Introdução**            | Uma visão geral sobre os componentes fundamentais das Redes Neurais.                                                  |
| **🔧 Perceptron**             | Explore interativamente o primeiro neurônio artificial, ajustando pesos e bias para entender suas limitações.           |
| **🌐 Redes Neurais (MLP)**   | Construa visualmente a arquitetura de uma rede multicamadas e explore as principais funções de ativação.              |
| **🔄 Backpropagation**      | Desvende o "coração" do aprendizado: um passo a passo detalhado e interativo do algoritmo.                             |
| **🧠 MLP em Ação**           | Veja na prática como um MLP resolve problemas não-lineares, com separação de dados de treino/teste e métricas de acurácia. |
| **🖼️ CNNs**                  | Aplique filtros (kernels) em uma imagem e veja como a operação de convolução extrai características visuais.           |
| **📜 RNNs**                  | Entenda o conceito de memória e processamento de sequências com ilustrações clássicas de RNNs, LSTMs e GRUs.            |
| **🤖 Transformers**          | Uma demonstração de ponta a ponta com um tokenizer pré-treinado para português, mostrando geração de texto antes e depois de um treinamento simulado. |
| **🎮 Playground Interativo** | Um sandbox completo para treinar seu próprio MLP, ajustando dataset, arquitetura e hiperparâmetros, e visualizando a fronteira de decisão resultante. |


## 📂 Estrutura do Projeto

O código é organizado de forma modular para facilitar a manutenção e a contribuição:

```
guia_redes_neurais/
├── 📂 assets/              # Imagens e outros recursos estáticos
├── main.py                # Ponto de entrada: roteador principal e UI global
├── requirements.txt       # Dependências do projeto
├── secao_*.py             # Cada seção do app em seu próprio módulo
└── visualizacoes.py       # Funções de plotagem reutilizáveis
```

- **`main.py`**: Configura a página, a barra de navegação e chama a função da seção apropriada.
- **`secao_*.py`**: Cada arquivo contém toda a lógica e os elementos de UI para uma única seção do aplicativo.

## 🛠️ Tecnologias Utilizadas

- **Framework Web:** [Streamlit](https://streamlit.io/)
- **Machine Learning & Data Science:** [Scikit-learn](https://scikit-learn.org/), [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers/index), [TensorFlow](https://www.tensorflow.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Visualização de Dados:** [Matplotlib](https://matplotlib.org/), [Plotly](https://plotly.com/), [Seaborn](https://seaborn.pydata.org/)
- **Processamento de Sinais:** [SciPy](https://scipy.org/)

## 🚀 Instalação e Execução

Siga os passos abaixo para executar o aplicativo localmente.

#### Pré-requisitos
- Python 3.10 ou superior
- Git

#### Passos

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Felipecataneo/redes-neurais.git
    cd SEU_REPOSITORIO
    ```

2.  **Crie e ative um ambiente virtual (Recomendado):**
    ```bash
    # Criar o ambiente
    python -m venv .venv

    # Ativar o ambiente
    # No Windows (PowerShell):
    .\.venv\Scripts\Activate.ps1
    # No macOS/Linux:
    source .venv/bin/activate
    ```
    *Alternativa rápida com [uv](https://github.com/astral-sh/uv): `uv venv`*

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Alternativa rápida com uv: `uv pip sync requirements.txt`*


4.  **Execute o aplicativo Streamlit:**
    ```bash
    streamlit run main.py
    ```
    O aplicativo será aberto automaticamente em uma nova aba do seu navegador.

## 🤝 Como Contribuir

Contribuições são sempre bem-vindas! Se você tiver sugestões, novos recursos ou correções, sinta-se à vontade para:

1.  Fazer um **Fork** deste repositório.
2.  Criar uma nova **Branch** (`git checkout -b minha-feature`).
3.  Fazer suas alterações e **Commit** (`git commit -m 'Adiciona nova feature'`).
4.  Fazer um **Push** para a sua Branch (`git push origin minha-feature`).
5.  Abrir um **Pull Request**.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
