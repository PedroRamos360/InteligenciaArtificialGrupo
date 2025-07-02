"""
Trabalho de Classificação de Doenças Renais Crônicas (CKD)
Análise Exploratória de Dados

Este módulo realiza a análise exploratória dos dados de CKD, incluindo:
- Estatísticas descritivas
- Visualizações dos dados
- Análise de correlações
- Identificação de valores faltantes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")


class CKDExploratoryAnalysis:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.prepare_data()

    def prepare_data(self):
        print("=== INFORMAÇÕES BÁSICAS DO DATASET ===")
        print(f"Dimensões do dataset: {self.df.shape}")
        print(f"Colunas: {list(self.df.columns)}")
        print("\n=== TIPOS DE DADOS ===")
        print(self.df.dtypes)

    def basic_statistics(self):
        print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print("\n--- Variáveis Numéricas ---")
        print(self.df[numeric_cols].describe())
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        print(f"\n--- Variáveis Categóricas ---")
        for col in categorical_cols:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts())
        print("\n=== VALORES FALTANTES ===")
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        missing_df = pd.DataFrame(
            {
                "Coluna": missing_values.index,
                "Valores Faltantes": missing_values.values,
                "Porcentagem (%)": missing_percent.values,
            }
        )
        missing_df = missing_df[missing_df["Valores Faltantes"] > 0].sort_values(
            "Valores Faltantes", ascending=False
        )
        if not missing_df.empty:
            print(missing_df)
        else:
            print("Não há valores faltantes no dataset.")
        return missing_df

    def target_analysis(self):
        print("\n=== ANÁLISE DAS VARIÁVEIS ALVO ===")
        print("\n--- CKD_Stage (Classificação Multiclasse) ---")
        stage_counts = self.df["CKD_Stage"].value_counts().sort_index()
        print(stage_counts)
        print(f"Distribuição percentual:")
        print((stage_counts / len(self.df) * 100).round(2))
        print("\n--- CKD_Progression (Classificação Binária) ---")
        progression_counts = self.df["CKD_Progression"].value_counts()
        print(progression_counts)
        print(f"Distribuição percentual:")
        print((progression_counts / len(self.df) * 100).round(2))
        return stage_counts, progression_counts

    def create_visualizations(self):
        graphs_dir = "graphs"
        os.makedirs(graphs_dir, exist_ok=True)
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        stage_counts = self.df["CKD_Stage"].value_counts().sort_index()
        axes[0].bar(stage_counts.index, stage_counts.values, color="skyblue", alpha=0.7)
        axes[0].set_title(
            "Distribuição dos Estágios de CKD", fontsize=14, fontweight="bold"
        )
        axes[0].set_xlabel("Estágio CKD")
        axes[0].set_ylabel("Frequência")
        axes[0].grid(True, alpha=0.3)
        progression_counts = self.df["CKD_Progression"].value_counts()
        labels = ["Não Progressão", "Progressão"]
        colors = ["lightgreen", "lightcoral"]
        axes[1].pie(
            progression_counts.values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1].set_title(
            "Distribuição da Progressão de CKD", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(graphs_dir, "target_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        numeric_cols = [
            "Age",
            "Systolic_Pressure",
            "BMI",
            "Hemoglobin",
            "Albumin",
            "Creatinine",
            "eGFR",
            "Protein_Creatinine_Ratio",
        ]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        for i, col in enumerate(numeric_cols):
            if col in self.df.columns:
                self.df[col].hist(bins=30, ax=axes[i], alpha=0.7, color="steelblue")
                axes[i].set_title(f"Distribuição de {col}", fontweight="bold")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Frequência")
                axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(graphs_dir, "numeric_distributions.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        numeric_data = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title(
            "Matriz de Correlação das Variáveis Numéricas",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(graphs_dir, "correlation_matrix.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        for i, col in enumerate(numeric_cols):
            if col in self.df.columns:
                self.df.boxplot(column=col, by="CKD_Stage", ax=axes[i])
                axes[i].set_title(f"{col} por Estágio CKD")
                axes[i].set_xlabel("Estágio CKD")
                axes[i].set_ylabel(col)
        plt.suptitle(
            "Distribuição das Variáveis Numéricas por Estágio CKD",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(graphs_dir, "boxplots_by_stage.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sex_progression = (
            pd.crosstab(self.df["Sex"], self.df["CKD_Progression"], normalize="index")
            * 100
        )
        sex_progression.plot(kind="bar", ax=axes[0], color=["lightgreen", "lightcoral"])
        axes[0].set_title("Progressão CKD por Sexo (%)")
        axes[0].set_xlabel("Sexo (1=Masculino, 2=Feminino)")
        axes[0].set_ylabel("Porcentagem")
        axes[0].legend(["Não Progressão", "Progressão"])
        axes[0].tick_params(axis="x", rotation=0)
        self.df["Age_Group"] = pd.cut(
            self.df["Age"],
            bins=[0, 40, 60, 80, 100],
            labels=["<40", "40-60", "60-80", ">80"],
        )
        age_progression = (
            pd.crosstab(
                self.df["Age_Group"], self.df["CKD_Progression"], normalize="index"
            )
            * 100
        )
        age_progression.plot(kind="bar", ax=axes[1], color=["lightgreen", "lightcoral"])
        axes[1].set_title("Progressão CKD por Faixa Etária (%)")
        axes[1].set_xlabel("Faixa Etária")
        axes[1].set_ylabel("Porcentagem")
        axes[1].legend(["Não Progressão", "Progressão"])
        axes[1].tick_params(axis="x", rotation=45)
        hyp_progression = (
            pd.crosstab(
                self.df["Hypertension"], self.df["CKD_Progression"], normalize="index"
            )
            * 100
        )
        hyp_progression.plot(kind="bar", ax=axes[2], color=["lightgreen", "lightcoral"])
        axes[2].set_title("Progressão CKD por Hipertensão (%)")
        axes[2].set_xlabel("Hipertensão (0=Não, 1=Sim)")
        axes[2].set_ylabel("Porcentagem")
        axes[2].legend(["Não Progressão", "Progressão"])
        axes[2].tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(
            os.path.join(graphs_dir, "progression_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def generate_report(self):
        print("INICIANDO ANÁLISE EXPLORATÓRIA DE DADOS...")
        print("=" * 60)
        missing_df = self.basic_statistics()
        stage_counts, progression_counts = self.target_analysis()
        self.create_visualizations()
        print("\n=== RESUMO DA ANÁLISE EXPLORATÓRIA ===")
        print(
            f"• Dataset com {self.df.shape[0]} registros e {self.df.shape[1]} variáveis"
        )
        print(f"• {len(missing_df)} variáveis com valores faltantes")
        print(f"• CKD_Stage: {len(stage_counts)} classes (estágios 1-5)")
        print(f"• CKD_Progression: 2 classes (0=Não, 1=Sim)")
        print(f"• Taxa de progressão: {(progression_counts[1]/len(self.df)*100):.1f}%")


if __name__ == "__main__":
    data_path = "ckd.csv"
    eda = CKDExploratoryAnalysis(data_path)
    eda.generate_report()
