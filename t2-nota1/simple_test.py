"""
Script Simplificado para Teste dos Modelos CKD

Este script executa apenas o treinamento básico dos modelos
sem otimização extensiva de hiperparâmetros, para demonstrar
a funcionalidade completa do trabalho.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from preprocessing import CKDPreprocessor
import warnings
import os

warnings.filterwarnings("ignore")


def train_simple_models(data):
    graphs_dir = "graphs"
    os.makedirs(graphs_dir, exist_ok=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_stage_train = data["y_stage_train"]
    y_stage_test = data["y_stage_test"]
    y_prog_train = data["y_prog_train"]
    y_prog_test = data["y_prog_test"]
    results = {}
    models = {}
    print("TREINAMENTO SIMPLIFICADO DOS MODELOS")
    print("=" * 50)
    model_configs = {
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, random_state=42
        ),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42
        ),
    }
    tasks = [
        ("binary", y_prog_train, y_prog_test, "CKD_Progression"),
        ("multiclass", y_stage_train, y_stage_test, "CKD_Stage"),
    ]
    for task_type, y_train, y_test, task_name in tasks:
        print(f"\n--- {task_name} ({task_type}) ---")
        for model_name, model in model_configs.items():
            print(f"\nTreinando {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            if task_type == "binary":
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, y_pred_proba, multi_class="ovr", average="weighted"
                )
            key = f"{model_name}_{task_type}"
            results[key] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_roc": auc,
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred),
            }
            models[key] = model
            print(f"  Acurácia: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC-ROC: {auc:.4f}")
            plt.figure(figsize=(6, 5))
            if task_type == "binary":
                class_names = ["Não Progressão", "Progressão"]
            else:
                class_names = [f"Estágio {i}" for i in sorted(y_test.unique())]
            sns.heatmap(
                confusion_matrix(y_test, y_pred),
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.title(f"Matriz de Confusão - {model_name} ({task_type})")
            plt.xlabel("Predito")
            plt.ylabel("Real")
            plt.tight_layout()
            filename = os.path.join(
                graphs_dir, f"confusion_simple_{model_name}_{task_type}.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

    return models, results



def main():
    print("TESTE SIMPLIFICADO - CLASSIFICAÇÃO CKD")
    print("=" * 50)
    data_path = "ckd.csv"
    save_path = "."
    try:
        preprocessor = CKDPreprocessor()
        preprocessor.load_preprocessors(save_path)
        print("Preprocessadores carregados com sucesso!")
        processed_data = preprocessor.full_preprocessing_pipeline(data_path)
    except:
        print("Processando dados do zero...")
        preprocessor = CKDPreprocessor()
        processed_data = preprocessor.full_preprocessing_pipeline(data_path, save_path)
    models, results = train_simple_models(processed_data)
    joblib.dump(models, f"{save_path}/simple_trained_models.pkl")
    joblib.dump(results, f"{save_path}/simple_model_results.pkl")

    print(f"\nTeste concluído! Arquivos salvos em {save_path}")


if __name__ == "__main__":
    main()
