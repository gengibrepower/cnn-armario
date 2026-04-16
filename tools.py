import os
from pathlib import Path

def listar_arquivos(base="."):
    """Lista todos os arquivos do projeto"""
    arquivos = []
    for root, dirs, files in os.walk(base):
        for file in files:
            arquivos.append(os.path.join(root, file))
    return arquivos


def contar_linhas(extensoes=None):
    """
    Conta linhas de código por extensão
    """
    if extensoes is None:
        extensoes = [".py", ".js", ".html", ".css"]

    resultado = {ext: 0 for ext in extensoes}

    for root, dirs, files in os.walk("."):
        for file in files:
            for ext in extensoes:
                if file.endswith(ext):
                    caminho = os.path.join(root, file)
                    try:
                        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                            linhas = len(f.readlines())
                            resultado[ext] += linhas
                    except Exception:
                        pass

    return resultado


def procurar_texto(texto, base="."):
    """
    Procura um texto dentro dos arquivos do projeto
    """
    resultados = []

    for root, dirs, files in os.walk(base):
        for file in files:
            caminho = os.path.join(root, file)

            try:
                with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                    for numero, linha in enumerate(f, start=1):
                        if texto in linha:
                            resultados.append((caminho, numero, linha.strip()))
            except Exception:
                continue

    return resultados


def tamanho_projeto(base="."):
    """
    Calcula o tamanho total do projeto
    """
    total = 0

    for root, dirs, files in os.walk(base):
        for file in files:
            caminho = os.path.join(root, file)
            try:
                total += os.path.getsize(caminho)
            except Exception:
                pass

    return total


def formatar_bytes(num):
    """
    Converte bytes para formato legível
    """
    for unidade in ["B", "KB", "MB", "GB"]:
        if num < 1024:
            return f"{num:.2f} {unidade}"
        num /= 1024
    return f"{num:.2f} TB"


def main():
    print("=== Ferramentas do Projeto ===\n")

    print("Arquivos encontrados:")
    arquivos = listar_arquivos()
    print(len(arquivos), "arquivos\n")

    print("Linhas por linguagem:")
    linhas = contar_linhas()
    for ext, qtd in linhas.items():
        print(f"{ext}: {qtd}")

    print("\nTamanho total do projeto:")
    tamanho = tamanho_projeto()
    print(formatar_bytes(tamanho))


if __name__ == "__main__":
    main()