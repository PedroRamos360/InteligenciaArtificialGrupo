"""
Script de Instalação das Dependências
Instala todas as bibliotecas necessárias para o projeto CKD
"""

import subprocess
import sys


def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} instalado com sucesso")
        return True
    except subprocess.CalledProcessError:
        print(f"Erro ao instalar {package}")
        return False


def main():
    print("=" * 60)
    print("INSTALAÇÃO DAS DEPENDÊNCIAS - PROJETO CKD")
    print("=" * 60)
    packages = [
        "pandas==2.0.3",
        "numpy==1.24.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "scikit-learn==1.3.0",
        "scipy==1.11.1",
        "plotly==5.15.0",
        "kaleido==0.2.1",
        "joblib",
    ]
    print(f"Instalando {len(packages)} pacotes...")
    print("-" * 40)
    print("Atualizando pip...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        )
        print("pip atualizado")
    except:
        print("Aviso: Não foi possível atualizar o pip")
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    print("-" * 40)
    if not failed_packages:
        print("Todas as dependências foram instaladas com sucesso!")
        print("\nPróximos passos:")
        print("1. Execute: python main.py")
        print("2. Aguarde a conclusão do pipeline")
        print("3. Verifique os arquivos gerados na pasta t2")
    else:
        print(f"{len(failed_packages)} pacote(s) falharam na instalação:")
        for package in failed_packages:
            print(f"  • {package}")
        print("\nTente instalar manualmente:")
        for package in failed_packages:
            print(f"  pip install {package}")


if __name__ == "__main__":
    main()
