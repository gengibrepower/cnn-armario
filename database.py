# database.py

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

# O caminho do banco de dados (SQLite)
ENGINE = create_engine("sqlite:///armario.db") 
Base = declarative_base()

# --- Modelos ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    nome = Column(String, unique=True, nullable=False)
    
    # 🛑 NOVIDADE: Campo para armazenar o HASH da senha
    password_hash = Column(String(128), nullable=False) 
    
    roupas = relationship("Roupa", back_populates="user", cascade="all, delete-orphan")

class Roupa(Base):
    __tablename__ = "clothes"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    path = Column(String, nullable=False)
    categoria = Column(String, nullable=False)
    criado_em = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="roupas")


# --- Funções de Inicialização e Sessão ---

def init_db():
    # Isso criará as tabelas se elas não existirem
    # 🛑 ATENÇÃO: Se a tabela 'users' já existir, você deve DELETAR o arquivo armario.db
    # para que a nova coluna 'password_hash' seja criada!
    Base.metadata.create_all(ENGINE) 
    print("Banco de dados inicializado.")

Session = sessionmaker(bind=ENGINE)