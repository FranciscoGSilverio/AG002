# 🧠 Classificação de Dados com Machine Learning

Este projeto treina um modelo de **Machine Learning** para classificar dados numéricos utilizando ferramentas do **scikit-learn**.  
O script principal realiza o pré-processamento dos dados, treina o modelo, avalia seu desempenho e salva o resultado para uso posterior.

---

## 📦 Estrutura do Projeto

```bash
project/
│
├── src/              # Funções utilitárias
│   ├── data_utils.py
│   ├── model_utils.py
│   └── __init__.py
│
├── train_model.py      # Script principal de treinamento
├── config.py           # Basic confugurations and constants
├── requirements.txt    # Dependências
└── README.md
```

## 🚀 Como Executar

### ✅ 1. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```
### ✅ 2. Instalar as dependências

```bash
pip install -r requirements.txt
```
### ✅ 3. Executar o treinamento do modelo

```bash
python train_model.py
```

