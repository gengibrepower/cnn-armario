# Armário Inteligente – README

Este projeto é um sistema web em Flask que permite:

- Cadastro e login de usuários
- Upload de imagens de roupas
- Classificação automática usando um modelo de IA em PyTorch
- Armazenamento das roupas no banco SQLite

## 📌 Estrutura necessária do projeto

```
/projeto
    app.py
    database.py
    predict.py
    train.py
    classes.json
    best_model.pth
    /templates
        index.html
        armario.html
    /uploads
    requirements.txt
    README.md
```

## 🛠 Tecnologias usadas

- Python 3.12
- Flask
- SQLAlchemy (SQLite)
- PyTorch + Torchvision
- Pillow
- JSON

## 📦 Instalação

### 1. Instalar dependências

```
pip install -r requirements.txt
```

Se usar PyTorch CPU:

```
pip install torch torchvision
```

### 2. Inicializar banco de dados

O banco será criado automaticamente ao rodar o app:

```
python app.py
```

## 🚀 Execução

```
python app.py
```

Acesse:

```
http://127.0.0.1:5000
```

## 📁 Arquivos importantes

### **database.py**
- Cria e gerencia o banco SQLite (`armario.db`)
- Define tabelas `users` e `clothes`

### **predict.py**
- Carrega modelo `best_model.pth`
- Aplica transformações
- Executa inferência para classificar roupas

### **train.py**
- Contém a arquitetura `ClothingCNNComplex`
- Necessário para carregar os pesos do PyTorch

### **classes.json**
- Mapeia IDs → nomes de categorias

### **best_model.pth**
- Pesos treinados do modelo

## ⚠️ Importante

Se modificar o modelo no `train.py`, talvez seja preciso **apagar o arquivo `armario.db`** para recriar as tabelas corretamente.

Se `best_model.pth` ou `classes.json` não estiverem presentes, o sistema funciona, mas a IA será desativada.

---

## 👤 Autor

Felipe Gegembauer
