# app.py (Código Completo com Segurança, Sessões e API de Categorias)

from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
from datetime import datetime
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from database import Session, User, Roupa, init_db
from predict import predict_category 
from sqlalchemy import func

# --- Configuração Flask e Pastas ---

BASE_DIR_UPLOADS = "uploads" 
# Cria a pasta 'uploads' se ela não existir
os.makedirs(BASE_DIR_UPLOADS, exist_ok=True)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# Chave secreta obrigatória para o funcionamento seguro das Sessões
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "uma_chave_secreta_padrao_muito_forte_12345")

# Inicializa as tabelas do DB ao iniciar a aplicação
init_db()


# --- Funções Auxiliares (Para evitar repetição de código) ---

def roupa_to_dict(roupa: Roupa):
    """Converte um objeto Roupa para um dicionário JSON amigável."""
    return {
        "id": roupa.id,
        "categoria": roupa.categoria,
        "image_url": f"/image/{roupa.path}",
        "criado_em": roupa.criado_em.isoformat() if roupa.criado_em else None
    }

def is_logged_in():
    """Verifica se o user_id está na sessão (usuário logado)."""
    return 'user_id' in session

# --- Rotas de Autenticação e Navegação ---

@app.route("/", methods=["GET"])
def login_page():
    """Rota inicial: Se logado, redireciona para o armário. Senão, mostra a tela de login (index.html)."""
    if is_logged_in():
        return redirect(url_for('armario_page'))
    # Renderiza o index.html, que agora é a tela de login
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    """Tenta autenticar o usuário pelo nome/ID e senha."""
    identificador = request.form.get("identificador")
    senha = request.form.get("senha") 

    if not identificador or not senha:
        return jsonify({"erro": "Forneça Nome/ID e Senha"}), 400

    try:
        user_id = int(identificador)
    except ValueError:
        user_id = None

    try:
        with Session() as session_db:
            user = None
            if user_id:
                user = session_db.get(User, user_id)

            if not user:
                user = session_db.query(User).filter(User.nome == identificador).first()

            if user:
                # Verifica se a senha fornecida corresponde ao hash armazenado
                if check_password_hash(user.password_hash, senha):
                    session['user_id'] = user.id
                    session['username'] = user.nome
                    return jsonify({"ok": True, "user_id": user.id, "nome": user.nome})
                else:
                    return jsonify({"erro": "Senha incorreta"}), 401 
            else:
                return jsonify({"erro": "Usuário não encontrado"}), 401 

    except Exception as e:
        logging.error(f"Erro no processo de login: {e}")
        return jsonify({"erro": "Erro interno de processamento."}), 500


@app.route("/logout", methods=["POST"])
def logout():
    """Remove o usuário da sessão."""
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({"ok": True, "mensagem": "Deslogado com sucesso."})


@app.route("/armario", methods=["GET"])
def armario_page():
    """Nova rota protegida: Armário Virtual do usuário logado (armario.html)."""
    if not is_logged_in():
        return redirect(url_for('login_page'))
    
    # Renderiza o armário, passando dados do usuário da sessão
    return render_template(
        "armario.html", 
        user_id=session['user_id'], 
        username=session['username']
    )


# --- Rotas da API (Protegidas) ---

@app.route("/register", methods=["POST"])
def register():
    """Registra um novo usuário com senha."""
    nome = request.form.get("nome")
    senha = request.form.get("senha") 

    if not nome or not senha:
        return jsonify({"erro": "Forneça nome e senha"}), 400

    try:
        # Cria o hash da senha de forma segura
        password_hash = generate_password_hash(senha)

        with Session() as session_db:
            # Verifica se o nome já existe
            if session_db.query(User).filter(User.nome == nome).first():
                return jsonify({"erro": "Nome de usuário já existe. Tente fazer login."}), 409 # 409 Conflict

            # Salva o hash, não a senha em texto puro
            user = User(nome=nome, password_hash=password_hash) 
            session_db.add(user)
            session_db.commit()
            
            return jsonify({"user_id": user.id, "nome": user.nome})
            
    except Exception as e:
        logging.error(f"Erro interno ao registrar usuário: {e}")
        return jsonify({"erro": "Erro interno ao salvar usuário no banco de dados."}), 500


@app.route("/upload-item", methods=["POST"])
def upload_item():
    """Faz o upload, classifica e salva no DB do usuário logado."""
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado. Faça login."}), 401
    
    user_id = session['user_id']

    if "file" not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"erro": "Arquivo não selecionado"}), 400
        
    ext = file.filename.split(".")[-1]
    filename = f"{int(datetime.now().timestamp())}_{user_id}.{ext}"
    filepath = os.path.join(BASE_DIR_UPLOADS, filename)

    try:
        file.save(filepath)
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo: {e}")
        return jsonify({"erro": "Falha ao salvar o arquivo"}), 500

    categoria = predict_category(filepath)
    
    try:
        with Session() as session_db:
            roupa = Roupa(
                user_id=user_id,
                path=filename,
                categoria=categoria,
            )

            session_db.add(roupa)
            session_db.commit()

            return jsonify(roupa_to_dict(roupa))
    except Exception as e:
        logging.error(f"Erro ao salvar roupa no banco de dados: {e}")
        return jsonify({"erro": "Erro interno ao salvar classificação no DB."}), 500


@app.route("/items/categories", methods=["GET"])
def get_categories():
    """Retorna uma lista de categorias únicas para o usuário logado."""
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado."}), 401
        
    user_id = session['user_id']

    with Session() as session_db:
        # Consulta para selecionar valores DISTINTOS da coluna 'categoria' para o usuário
        categories = session_db.query(Roupa.categoria)\
                                .filter(Roupa.user_id == user_id)\
                                .distinct()\
                                .all()
        
        # Converte a lista de tuplas em uma lista simples de strings
        category_list = [c[0] for c in categories]
        
        return jsonify(category_list)


@app.route("/items", methods=["GET"])
def listar_items():
    """Lista todos os itens do usuário logado, com filtro opcional por 'categoria'."""
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado. Faça login."}), 401
        
    user_id = session['user_id']
    categoria_filtro = request.args.get('categoria') 
    
    with Session() as session_db:
        query = session_db.query(Roupa).filter(Roupa.user_id == user_id)
        
        if categoria_filtro:
            # Busca parcial e insensível a maiúsculas/minúsculas
            query = query.filter(Roupa.categoria.ilike(f"%{categoria_filtro}%")) 
            
        data = query.all()
        
        return jsonify([roupa_to_dict(r) for r in data])


@app.route("/items/<int:item_id>", methods=["DELETE"])
def deletar(item_id):
    """Deleta um item e seu arquivo de imagem, garantindo que pertença ao usuário logado."""
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado. Faça login."}), 401
    
    user_id = session['user_id']
    
    with Session() as session_db:
        # Busca o item usando o id e o user_id (segurança)
        roupa = session_db.query(Roupa).filter(
            Roupa.id == item_id, 
            Roupa.user_id == user_id
        ).first()

        if not roupa:
            return jsonify({"erro": "Item não encontrado ou não pertence ao usuário"}), 404

        filepath = os.path.join(BASE_DIR_UPLOADS, roupa.path)
        
        try:
            os.remove(filepath)
            logging.info(f"Arquivo {roupa.path} removido.")
        except FileNotFoundError:
            logging.warning(f"Arquivo {roupa.path} não encontrado em disco.")
        except Exception as e:
            logging.error(f"Erro ao remover arquivo {filepath}: {e}")

        session_db.delete(roupa)
        session_db.commit()

        return jsonify({"ok": True, "item_id": item_id})


@app.route("/image/<filename>")
def image(filename):
    """Serve o arquivo de imagem estático da pasta uploads."""
    return send_from_directory(BASE_DIR_UPLOADS, filename)

@app.route("/items/count", methods=["GET"])
def get_item_count():
    """Retorna a contagem de itens por categoria para o usuário logado."""
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado."}), 401
        
    user_id = session['user_id']

    with Session() as session_db:
        # Consulta que agrupa e conta itens por categoria
        counts = session_db.query(Roupa.categoria, func.count(Roupa.id))\
                           .filter(Roupa.user_id == user_id)\
                           .group_by(Roupa.categoria)\
                           .all()
        
        # Converte a lista de tuplas em um dicionário para fácil uso no JS
        count_dict = {categoria: count for categoria, count in counts}
        
        return jsonify(count_dict)


if __name__ == "__main__":
    app.run(debug=True, port=8000)