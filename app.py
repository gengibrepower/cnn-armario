from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
from datetime import datetime
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func
from sqlalchemy.sql.expression import func as sql_func
from database import Session, User, Roupa, init_db
from predict import predict_category # Importa a funcao de predicao de ML

# --- Configuracao Flask e Pastas ---

BASE_DIR_UPLOADS = "uploads" 
# Cria a pasta 'uploads' se ela nao existir
os.makedirs(BASE_DIR_UPLOADS, exist_ok=True)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# Chave secreta obrigatoria para o funcionamento seguro das Sessoes
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "uma_chave_secreta_padrao_muito_forte_12345")

# Inicializa as tabelas do DB ao iniciar a aplicacao
init_db()


# --- Funcoes Auxiliares ---
def roupa_to_dict(roupa: Roupa):
    # Converte um objeto Roupa para um dicionario JSON amigavel.
    return {
        "id": roupa.id,
        "categoria": roupa.categoria,
        # Garante que a URL e /uploads/ que corresponde a rota de servico
        "image_url": f"/uploads/{roupa.path}", 
        "criado_em": roupa.criado_em.isoformat() if roupa.criado_em else None
    }

def is_logged_in():
    # Verifica se o user_id esta na sessao (usuario logado).
    return 'user_id' in session

# --- Rotas de Autenticacao e Navegacao ---

@app.route("/", methods=["GET"])
def login_page():
    if is_logged_in():
        return redirect(url_for('armario_page'))
    # Renderiza o index.html, que agora e a tela de login
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    identificador = request.form.get("identificador")
    senha = request.form.get("senha") 

    if not identificador or not senha:
        return jsonify({"erro": "Forneca Nome/ID e Senha"}), 400

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
                return jsonify({"erro": "Usuario nao encontrado"}), 401 

    except Exception as e:
        logging.error(f"Erro no processo de login: {e}")
        return jsonify({"erro": "Erro interno de processamento."}), 500


@app.route("/logout", methods=["POST"])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({"ok": True, "mensagem": "Deslogado com sucesso."})


@app.route("/armario", methods=["GET"])
def armario_page():
    # Armario Virtual do usuario logado (armario.html).
    if not is_logged_in():
        return redirect(url_for('login_page'))
    
    # Renderiza o armario, passando dados do usuario da sessao
    return render_template(
        "armario.html", 
        user_id=session['user_id'], 
        username=session['username']
    )


# --- Rotas da API (Protegidas) ---

@app.route("/register", methods=["POST"])
def register():
    nome = request.form.get("nome")
    senha = request.form.get("senha") 

    if not nome or not senha:
        return jsonify({"erro": "Forneca nome e senha"}), 400

    try:
        # Cria o hash da senha de forma segura
        password_hash = generate_password_hash(senha)

        with Session() as session_db:
            # Verifica se o nome ja existe
            if session_db.query(User).filter(User.nome == nome).first():
                return jsonify({"erro": "Nome de usuario ja existe. Tente fazer login."}), 409 # 409 Conflito

            # Salva o hash, nao a senha em texto puro
            user = User(nome=nome, password_hash=password_hash) 
            session_db.add(user)
            session_db.commit()
            
            return jsonify({"user_id": user.id, "nome": user.nome})
            
    except Exception as e:
        logging.error(f"Erro interno ao registrar usuario: {e}")
        return jsonify({"erro": "Erro interno ao salvar usuario no banco de dados."}), 500


@app.route("/upload-item", methods=["POST"])
def upload_item():
    # Faz o upload, classifica e salva no DB do usuario logado.
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado. Faca login."}), 401
    
    user_id = session['user_id']

    if "file" not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"erro": "Arquivo nao selecionado"}), 400
        
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
        return jsonify({"erro": "Erro interno ao salvar classificacao no DB."}), 500


@app.route("/items/categories", methods=["GET"])
def get_categories():
    # Retorna uma lista de categorias unicas para o usuario logado.
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado."}), 401
        
    user_id = session['user_id']

    with Session() as session_db:
        # Consulta para selecionar valores DISTINTOS da coluna 'categoria' para o usuario
        categories = session_db.query(Roupa.categoria)\
                             .filter(Roupa.user_id == user_id)\
                             .distinct()\
                             .all()
        
        # Converte a lista de tuplas em uma lista simples de strings
        category_list = [c[0] for c in categories]
        
        return jsonify(category_list)


@app.route("/items", methods=["GET"])
def listar_items():
    # Lista todos os itens do usuario logado, com filtro opcional por 'categoria'.
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado. Faca login."}), 401
        
    user_id = session['user_id']
    categoria_filtro = request.args.get('categoria') 
    
    with Session() as session_db:
        query = session_db.query(Roupa).filter(Roupa.user_id == user_id)
        
        if categoria_filtro:
            # Busca parcial e insensivel a maiusculas/minusculas
            query = query.filter(Roupa.categoria.ilike(f"%{categoria_filtro}%")) 
            
        data = query.all()
        
        return jsonify([roupa_to_dict(r) for r in data])


@app.route("/items/<int:item_id>", methods=["DELETE"])
def deletar(item_id):
    # Deleta um item e seu arquivo de imagem, garantindo que pertença ao usuario logado.
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado. Faca login."}), 401
    
    user_id = session['user_id']
    
    with Session() as session_db:
        # Busca o item usando o id e o user_id (seguranca)
        roupa = session_db.query(Roupa).filter(
            Roupa.id == item_id, 
            Roupa.user_id == user_id
        ).first()

        if not roupa:
            return jsonify({"erro": "Item nao encontrado ou nao pertence ao usuario"}), 404

        filepath = os.path.join(BASE_DIR_UPLOADS, roupa.path)
        
        try:
            os.remove(filepath)
            logging.info(f"Arquivo {roupa.path} removido.")
        except FileNotFoundError:
            logging.warning(f"Arquivo {roupa.path} nao encontrado em disco.")
        except Exception as e:
            logging.error(f"Erro ao remover arquivo {filepath}: {e}")

        session_db.delete(roupa)
        session_db.commit()

        return jsonify({"ok": True, "item_id": item_id})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    # Serve o arquivo de imagem estatico da pasta uploads
    full_dir_path = os.path.join(app.root_path, BASE_DIR_UPLOADS)
    return send_from_directory(full_dir_path, filename)

@app.route("/items/count", methods=["GET"])
def get_item_count():
    # Retorna a contagem de itens por categoria para o usuario logado.
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado."}), 401
        
    user_id = session['user_id']

    with Session() as session_db:
        # Consulta que agrupa e conta itens por categoria
        counts = session_db.query(Roupa.categoria, func.count(Roupa.id))\
                           .filter(Roupa.user_id == user_id)\
                           .group_by(Roupa.categoria)\
                           .all()
        
        # Converte a lista de tuplas em um dicionario
        count_dict = {categoria: count for categoria, count in counts}
        
        return jsonify(count_dict)


@app.route("/outfit/generate", methods=["GET"])
def generate_outfit():
    # Gera um look aleatorio pegando um item de cada 'zona' de vestuario.
    if not is_logged_in():
        return jsonify({"erro": "Acesso negado."}), 401
        
    user_id = session['user_id']
    
    # Define as "zonas" do outfit e quais categorias de ML pertencem a elas.
    OUTFIT_ZONES = {
        "parte_cima": ["T-Shirt", "Shirt", "Blouse", "Polo", "Longsleeve", "Hoodie", "Top", "Undershirt"],
        "cobertura": ["Blazer", "Outwear"], 
        "parte_baixo": ["Pants", "Shorts", "Skirt"],
        "calcado": ["Shoes"],
    }
    
    outfit = {}
    
    try:
        with Session() as session_db:
            
            # Para cada zona, tenta pegar uma peca aleatoria
            for zone, categories in OUTFIT_ZONES.items():
                
                # 1. Filtra por user_id E pelas categorias da zona
                # 2. Ordena aleatoriamente (ORDER BY RANDOM() ou sql_func.random())
                # 3. Limita a 1 resultado
                item = session_db.query(Roupa)\
                                 .filter(Roupa.user_id == user_id)\
                                 .filter(Roupa.categoria.in_(categories))\
                                 .order_by(sql_func.random())\
                                 .limit(1)\
                                 .first()
                
                if item:
                    # Adiciona a peca formatada ao dicionario do outfit
                    outfit[zone] = roupa_to_dict(item)

            if not outfit:
                return jsonify({"erro": "Seu armario esta vazio. Adicione mais pecas!"}), 404

            # Verifica se pelo menos o minimo (Cima e Baixo) foi montado
            if 'parte_cima' not in outfit or 'parte_baixo' not in outfit:
                return jsonify({"erro": "Nao foi possivel montar uma combinacao completa (Falta cima ou baixo)."}), 404


            return jsonify({
                "ok": True, 
                "outfit": outfit
            })

    except Exception as e:
        logging.error(f"Erro ao gerar outfit: {e}")
        return jsonify({"erro": "Erro interno ao processar a combinacao."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)