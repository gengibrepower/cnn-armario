# app.py (Com Login/Sessões)

from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
from datetime import datetime
import os
import logging
# Importa a lógica do banco de dados
from database import Session, User, Roupa, init_db
# Importa a função de predição de ML
from predict import predict_category 

# --- Configuração Flask e Pastas ---

BASE_DIR_UPLOADS = "uploads" 
os.makedirs(BASE_DIR_UPLOADS, exist_ok=True)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# 🛑 NOVIDADE: Adiciona uma chave secreta para a sessão (MUITO IMPORTANTE!)
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

# Função de verificação para rotas protegidas
def is_logged_in():
    return 'user_id' in session

# --- Rotas de Autenticação e Navegação ---

@app.route("/", methods=["GET"])
def login_page():
    """Rota inicial: Se logado, redireciona para o armário. Senão, mostra a tela de login."""
    if is_logged_in():
        return redirect(url_for('armario_page'))
    # Renderiza o index.html, que agora será a tela de login
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    """Tenta autenticar o usuário pelo nome (ou ID, se for número)."""
    identificador = request.form.get("identificador") # Pode ser nome ou ID

    if not identificador:
        return jsonify({"erro": "Forneça seu Nome ou ID"}), 400

    try:
        user_id = int(identificador)
    except ValueError:
        user_id = None # Não é um ID numérico, tentaremos buscar pelo nome

    with Session() as session_db:
        user = None
        if user_id:
            user = session_db.get(User, user_id)
        
        # Se não encontrou pelo ID ou o identificador era um nome
        if not user:
            user = session_db.query(User).filter(User.nome == identificador).first()

        if user:
            # 🛑 SUCESSO: Armazena o ID na sessão
            session['user_id'] = user.id
            session['username'] = user.nome
            return jsonify({"ok": True, "user_id": user.id, "nome": user.nome})
        else:
            return jsonify({"erro": "Usuário não encontrado"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    """Remove o usuário da sessão."""
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({"ok": True, "mensagem": "Deslogado com sucesso."})


@app.route("/armario", methods=["GET"])
def armario_page():
    """Nova rota protegida: Armário Virtual do usuário logado."""
    if not is_logged_in():
        return redirect(url_for('login_page'))
    
    # Renderiza o armário, passando dados do usuário
    return render_template(
        "armario.html", 
        user_id=session['user_id'], 
        username=session['username']
    )


# --- Rotas da API (Protegidas) ---

@app.route("/register", methods=["POST"])
def register():
    """Registra um novo usuário."""
    # ... (código existente) ...

@app.route("/upload-item", methods=["POST"]) # 🛑 ROTA SIMPLIFICADA (sem user_id na URL)
def upload_item():
    """Faz o upload, classifica e salva no DB do usuário logado."""
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado. Faça login."}), 401
    
    user_id = session['user_id']
    # ... (código que salva o arquivo e classifica) ...
    # OBS: O bloco de código abaixo deve ser atualizado.

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
    
    with Session() as session_db:
        # Não precisa verificar o user, pois ele já está na sessão
        roupa = Roupa(
            user_id=user_id,
            path=filename,
            categoria=categoria,
        )

        session_db.add(roupa)
        session_db.commit()

        return jsonify(roupa_to_dict(roupa))


@app.route("/items", methods=["GET"]) # 🛑 ROTA SIMPLIFICADA (sem user_id na URL)
def listar_items():
    """Lista todos os itens do usuário logado, com filtro opcional por 'categoria'."""
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado. Faça login."}), 401
        
    user_id = session['user_id']
    categoria_filtro = request.args.get('categoria') 
    
    with Session() as session_db:
        query = session_db.query(Roupa).filter(Roupa.user_id == user_id)
        
        if categoria_filtro:
            query = query.filter(Roupa.categoria.ilike(f"%{categoria_filtro}%")) # Busca parcial (mais amigável)
            
        data = query.all()
        
        return jsonify([roupa_to_dict(r) for r in data])


@app.route("/items/<int:item_id>", methods=["DELETE"]) # 🛑 ROTA SIMPLIFICADA (sem user_id na URL)
def deletar(item_id):
    """Deleta um item e seu arquivo de imagem."""
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
    """Serve o arquivo de imagem estático."""
    return send_from_directory(BASE_DIR_UPLOADS, filename)


if __name__ == "__main__":
    app.run(debug=True, port=8000)