# database.py

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base

# Configuração do Banco de Dados
DATABASE_URL = "sqlite:///armario.db"
ENGINE = create_engine(DATABASE_URL, pool_recycle=3600)
Session = sessionmaker(bind=ENGINE)
Base = declarative_base()

# --- Modelos ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    nome = Column(String, nullable=False) # Adicionado nullable=False para campos obrigatórios

class Roupa(Base):
    __tablename__ = "clothes"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    path = Column(String, nullable=False)
    categoria = Column(String)
    criado_em = Column(DateTime, default=datetime.utcnow) 

# Função para criar as tabelas (chamada no app.py ou main.py)
def init_db():
    """Cria todas as tabelas no banco de dados, se não existirem."""
    Base.metadata.create_all(ENGINE)