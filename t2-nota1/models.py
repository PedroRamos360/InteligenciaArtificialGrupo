"""
Módulo de Modelos de Classificação para CKD

Este módulo implementa três modelos de classificação:
1. Árvores de Decisão (Decision Tree)
2. Redes Neurais Artificiais (MLP)
3. Random Forest (técnica adicional escolhida)

Cada modelo é treinado para ambos os problemas:
- CKD_Stage (classificação multiclasse)
- CKD_Progression (classificação binária)
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os

warnings.filterwarnings("ignore")


class CKDClassificationModels:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_params = {}

    def create_decision_tree(self, task_type="binary"):
        if task_type == "binary":
            param_grid = {
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 4],
                "criterion": ["gini"],
            }
        else:
            param_grid = {
                "max_depth": [7, 15, None],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 4],
                "criterion": ["gini"],
            }

        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            dt, param_grid, cv=3, scoring="f1_weighted", n_jobs=1
        )

        return grid_search

    def create_mlp(self, task_type="binary"):
        if task_type == "binary":
            param_grid = {
                "hidden_layer_sizes": [(50,), (100,)],
                "activation": ["relu"],
                "alpha": [0.01, 0.1],
                "learning_rate": ["adaptive"],
                "max_iter": [1000],
            }
        else:
            param_grid = {
                "hidden_layer_sizes": [(100,), (150,)],
                "activation": ["relu"],
                "alpha": [0.01, 0.1],
                "learning_rate": ["adaptive"],
                "max_iter": [1000],
            }

        mlp = MLPClassifier(random_state=42)
        grid_search = GridSearchCV(
            mlp, param_grid, cv=3, scoring="f1_weighted", n_jobs=1
        )

        return grid_search

    def create_random_forest(self, task_type="binary"):
        if task_type == "binary":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, None],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 4],
            }
        else:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [15, None],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 4],
            }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring="f1_weighted", n_jobs=1
        )

        return grid_search

    def train_model(self, model, X_train, y_train, model_name, task_type):
        print(f"\nTreinando {model_name} para {task_type}...")

        model.fit(X_train, y_train)

        self.best_params[f"{model_name}_{task_type}"] = model.best_params_

        print(f"Melhores parâmetros para {model_name} ({task_type}):")
        for param, value in model.best_params_.items():
            print(f"  {param}: {value}")

        return model.best_estimator_

    def evaluate_model(
        self, model, X_test, y_test, model_name, task_type, class_names=None
    ):
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

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
        }

        self.results[f"{model_name}_{task_type}"] = metrics

        print(f"\n=== RESULTADOS - {model_name} ({task_type}) ===")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")

        return metrics

    def plot_confusion_matrix(self, cm, model_name, task_type, class_names=None):
        graphs_dir = "graphs"
        os.makedirs(graphs_dir, exist_ok=True)

        plt.figure(figsize=(8, 6))

        if class_names is None:
            if task_type == "binary":
                class_names = ["Não Progressão", "Progressão"]
            else:
                class_names = [f"Estágio {i}" for i in range(1, len(cm) + 1)]

        sns.heatmap(
            cm,
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
            graphs_dir, f"confusion_matrix_{model_name}_{task_type}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_roc_curve(self, models_data, task_type):
        graphs_dir = "graphs"
        os.makedirs(graphs_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))

        for model, X_test, y_test, name in models_data:
            if task_type == "binary":
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc = roc_auc_score(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
            else:
                y_pred_proba = model.predict_proba(X_test)
                auc = roc_auc_score(
                    y_test, y_pred_proba, multi_class="ovr", average="weighted"
                )
                plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
                plt.text(
                    0.6,
                    0.4 - len(models_data) * 0.05,
                    f"{name}: AUC = {auc:.3f}",
                    fontsize=10,
                )

        if task_type == "binary":
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlabel("Taxa de Falsos Positivos")
            plt.ylabel("Taxa de Verdadeiros Positivos")
            plt.title(f"Curvas ROC - {task_type}")
            plt.legend()
        else:
            plt.xlabel("Comparação AUC")
            plt.ylabel("Modelos")
            plt.title(f"Comparação AUC - {task_type}")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = os.path.join(graphs_dir, f"roc_curves_{task_type}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

    def train_and_evaluate_all_models(self, data):
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_stage_train = data["y_stage_train"]
        y_stage_test = data["y_stage_test"]
        y_prog_train = data["y_prog_train"]
        y_prog_test = data["y_prog_test"]
        trained_models = {}
        print("INICIANDO TREINAMENTO DOS MODELOS...")
        print("=" * 60)
        tasks = [
            ("binary", y_prog_train, y_prog_test, "CKD_Progression"),
            ("multiclass", y_stage_train, y_stage_test, "CKD_Stage"),
        ]
        for task_type, y_train, y_test, task_name in tasks:
            print(f"\n{'='*40}")
            print(f"TAREFA: {task_name} ({task_type})")
            print(f"{'='*40}")
            dt_model = self.create_decision_tree(task_type)
            dt_trained = self.train_model(
                dt_model, X_train, y_train, "Decision_Tree", task_type
            )
            dt_metrics = self.evaluate_model(
                dt_trained, X_test, y_test, "Decision_Tree", task_type
            )
            self.plot_confusion_matrix(
                dt_metrics["confusion_matrix"], "Decision_Tree", task_type
            )
            trained_models[f"decision_tree_{task_type}"] = dt_trained
            mlp_model = self.create_mlp(task_type)
            mlp_trained = self.train_model(
                mlp_model, X_train, y_train, "MLP", task_type
            )
            mlp_metrics = self.evaluate_model(
                mlp_trained, X_test, y_test, "MLP", task_type
            )
            self.plot_confusion_matrix(
                mlp_metrics["confusion_matrix"], "MLP", task_type
            )
            trained_models[f"mlp_{task_type}"] = mlp_trained
            rf_model = self.create_random_forest(task_type)
            rf_trained = self.train_model(
                rf_model, X_train, y_train, "Random_Forest", task_type
            )
            rf_metrics = self.evaluate_model(
                rf_trained, X_test, y_test, "Random_Forest", task_type
            )
            self.plot_confusion_matrix(
                rf_metrics["confusion_matrix"], "Random_Forest", task_type
            )
            trained_models[f"random_forest_{task_type}"] = rf_trained
            models_data = [
                (dt_trained, X_test, y_test, "Decision Tree"),
                (mlp_trained, X_test, y_test, "MLP"),
                (rf_trained, X_test, y_test, "Random Forest"),
            ]
            self.plot_roc_curve(models_data, task_type)

        return trained_models

    def generate_comparison_report(self):
        print("\n" + "=" * 80)
        print("RELATÓRIO COMPARATIVO DOS MODELOS")
        print("=" * 80)

        results_data = []
        for model_task, metrics in self.results.items():
            model_name, task_type = model_task.rsplit("_", 1)
            results_data.append(
                {
                    "Modelo": model_name,
                    "Tarefa": task_type,
                    "Acurácia": metrics["accuracy"],
                    "Precisão": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1_score"],
                    "AUC-ROC": metrics["auc_roc"],
                }
            )

        results_df = pd.DataFrame(results_data)
        print("\n=== TABELA COMPARATIVA ===")
        print(results_df.to_string(index=False, float_format="%.4f"))
        results_df.to_csv("model_comparison.csv", index=False)
        for task in ["binary", "multiclass"]:
            task_results = results_df[results_df["Tarefa"] == task]
            if not task_results.empty:
                print(f"\n=== MELHOR MODELO PARA {task.upper()} ===")
                best_f1 = task_results.loc[task_results["F1-Score"].idxmax()]
                print(f"Modelo: {best_f1['Modelo']}")
                print(f"F1-Score: {best_f1['F1-Score']:.4f}")
                print(f"AUC-ROC: {best_f1['AUC-ROC']:.4f}")
        self.plot_model_comparison(results_df)

        return results_df

    def plot_model_comparison(self, results_df):
        graphs_dir = "graphs"
        os.makedirs(graphs_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        metrics = ["Acurácia", "Precisão", "Recall", "F1-Score", "AUC-ROC"]
        for i, task in enumerate(["binary", "multiclass"]):
            task_data = results_df[results_df["Tarefa"] == task]
            x = np.arange(len(metrics))
            width = 0.25
            for j, model in enumerate(task_data["Modelo"].unique()):
                model_data = task_data[task_data["Modelo"] == model]
                values = [model_data[metric].values[0] for metric in metrics]
                axes[i].bar(x + j * width, values, width, label=model, alpha=0.8)
            axes[i].set_xlabel("Métricas")
            axes[i].set_ylabel("Valores")
            axes[i].set_title(f"Comparação de Modelos - {task.upper()}")
            axes[i].set_xticks(x + width)
            axes[i].set_xticklabels(metrics, rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(graphs_dir, "model_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def save_models(self, models, save_path):
        joblib.dump(models, f"{save_path}/trained_models.pkl")
        joblib.dump(self.results, f"{save_path}/model_results.pkl")
        joblib.dump(self.best_params, f"{save_path}/best_parameters.pkl")
        print(f"Modelos e resultados salvos em {save_path}")


if __name__ == "__main__":
    pass
