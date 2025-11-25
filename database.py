from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

# O caminho do banco de dados (SQLite)t
ENGINE = create_engine("sqlite:///armario.db") 
Base = declarative_base()

# --- Modelos ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    nome = Column(String, unique=True, nullable=False)
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



def init_db():
    # Isso criara as tabelas se elas nao existirem
    # ATENCAO: Se a tabela 'users' ja existir, voce deve DELETAR o arquivo armario.db
    # para que a nova coluna 'password_hash' seja criada!
    Base.metadata.create_all(ENGINE) 
    print("Banco de dados inicializado.")

Session = sessionmaker(bind=ENGINE)