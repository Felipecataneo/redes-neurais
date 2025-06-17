🧠 Guia Avançado de Redes Neurais
![alt text](https://img.shields.io/badge/Python-3.10+-blue.svg)

![alt text](https://img.shields.io/badge/License-MIT-green.svg)

![alt text](https://img.shields.io/badge/Streamlit-1.35+-red.svg)
Este repositório contém o código-fonte de um aplicativo web interativo, desenvolvido com Streamlit, para servir como um guia didático sobre Redes Neurais. A ferramenta foi projetada especialmente para estudantes de pós-graduação, pesquisadores e entusiastas que desejam aprofundar seus conhecimentos em Machine Learning de forma visual e prática.

✨ Principais Recursos
O aplicativo é dividido em seções modulares, cobrindo desde os conceitos mais básicos até arquiteturas avançadas:
🔍 Introdução: Uma visão geral sobre o que são Redes Neurais e sua inspiração biológica.
🔧 O Perceptron: Explore o bloco de construção fundamental das redes neurais, suas capacidades e limitações.
🏛️ Arquitetura de Redes Neurais: Entenda como múltiplos neurônios são organizados em camadas para formar um Multilayer Perceptron (MLP).
🧠 MLP em Ação: Veja na prática como um MLP resolve problemas não-linearmente separáveis que o Perceptron não consegue.
🔄 Backpropagation: Uma explicação didática do algoritmo de aprendizado que treina as redes neurais.
🖼️ Redes Neurais Convolucionais (CNN): Demonstrações interativas das operações de convolução e pooling, essenciais para o processamento de imagens.
📜 Redes Neurais Recorrentes (RNN): Conceitos de memória e processamento de sequências, com foco em LSTM e GRU.
🎮 Playground Interativo: Um ambiente sandbox para treinar seu próprio MLP, ajustando dados, arquitetura e hiperparâmetros, e visualizando a fronteira de decisão resultante.
🛠️ Tecnologias Utilizadas
Framework Web: Streamlit
Machine Learning: Scikit-learn, TensorFlow
Manipulação de Dados: NumPy, Pandas
Visualização de Dados: Matplotlib, Plotly
Processamento de Sinais: SciPy
🚀 Instalação e Execução
Siga os passos abaixo para executar o aplicativo localmente.
Pré-requisitos
Python 3.10 ou superior
Git
Passos
Clone o repositório:
Generated bash
git clone https://github.com/Felipecataneo/redes-neurais
cd redes-neurais-main
Crie e ative um ambiente virtual:
É uma forte recomendação usar um ambiente virtual para isolar as dependências do projeto.
Usando venv (padrão do Python):
Generated bash
# Criar o ambiente
python -m venv .venv

# Ativar o ambiente
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No macOS/Linux:
source .venv/bin/activate
Use code with caution.
Bash
Usando uv (alternativa rápida):
Generated bash
# Criar o ambiente
uv venv

# Ativar o ambiente
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No macOS/Linux:
source .venv/bin/activate
Use code with caution.
Bash
Instale as dependências:
O arquivo requirements.txt contém todos os pacotes necessários com versões compatíveis.
Generated bash
pip install -r requirements.txt
Use code with caution.
Bash
Ou, se estiver usando uv:
Generated bash
uv pip sync requirements.txt
Use code with caution.
Bash
Execute o aplicativo Streamlit:
Com o ambiente ativado e as dependências instaladas, inicie o aplicativo:
Generated bash
streamlit run main.py
Use code with caution.
Bash
O aplicativo será aberto automaticamente em uma nova aba do seu navegador.
🤝 Como Contribuir
Contribuições são sempre bem-vindas! Se você tiver sugestões de melhoria, novos recursos ou correções de bugs, sinta-se à vontade para:
Fazer um Fork deste repositório.
Criar uma nova Branch (git checkout -b minha-feature).
Fazer suas alterações e Commit (git commit -m 'Adiciona nova feature').
Fazer um Push para a sua Branch (git push origin minha-feature).
Abrir um Pull Request.
📄 Licença
Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.