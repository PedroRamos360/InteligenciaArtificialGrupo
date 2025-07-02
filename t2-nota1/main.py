"""
Script Principal - Trabalho de Classificação CKD

Este script executa todo o pipeline do trabalho:
1. Análise Exploratória de Dados
2. Pré-processamento
3. Treinamento dos Modelos
4. Avaliação e Comparação
5. Geração do Relatório Final
"""

import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exploratory_analysis import CKDExploratoryAnalysis
from preprocessing import CKDPreprocessor
from models import CKDClassificationModels


def main():
    print("=" * 80)
    print("TRABALHO DE CLASSIFICAÇÃO DE DOENÇAS RENAIS CRÔNICAS (CKD)")
    print("=" * 80)
    print(f"Início da execução: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 80)
    data_path = "ckd.csv"
    save_path = "."
    try:
        print("\nETAPA 1: ANÁLISE EXPLORATÓRIA DE DADOS")
        print("-" * 50)
        start_time = time.time()
        eda = CKDExploratoryAnalysis(data_path)
        eda.generate_report()
        print(
            f"Análise exploratória concluída em {time.time() - start_time:.2f} segundos"
        )
        print("\nETAPA 2: PRÉ-PROCESSAMENTO DOS DADOS")
        print("-" * 50)
        start_time = time.time()
        preprocessor = CKDPreprocessor()
        processed_data = preprocessor.full_preprocessing_pipeline(data_path, save_path)
        print(f"Pré-processamento concluído em {time.time() - start_time:.2f} segundos")
        print("\nETAPA 3: TREINAMENTO E AVALIAÇÃO DOS MODELOS")
        print("-" * 50)
        start_time = time.time()
        classifier = CKDClassificationModels()
        trained_models = classifier.train_and_evaluate_all_models(processed_data)
        print(
            f"Treinamento dos modelos concluído em {time.time() - start_time:.2f} segundos"
        )
        print("\nETAPA 4: RELATÓRIO COMPARATIVO")
        print("-" * 50)
        start_time = time.time()
        results_df = classifier.generate_comparison_report()
        print(
            f"Relatório comparativo gerado em {time.time() - start_time:.2f} segundos"
        )
        print("\nETAPA 5: SALVANDO RESULTADOS")
        print("-" * 50)
        classifier.save_models(trained_models, save_path)
        print("\nETAPA 6: DISCUSSÃO E CONCLUSÕES")
        print("-" * 50)
        generate_final_discussion(results_df, classifier.results)
        print("\n" + "=" * 80)
        print("=" * 80)
        print(f"Fim da execução: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

        print("\nARQUIVOS GERADOS:")
        print("• Gráficos de análise exploratória (graphs/*.png)")
        print("• Tabela comparativa (model_comparison.csv)")
        print("• Modelos treinados (trained_models.pkl)")
        print("• Resultados detalhados (model_results.pkl)")
        print("• Preprocessadores (preprocessors.pkl)")

    except Exception as e:
        print(f"\nERRO durante a execução: {str(e)}")
        import traceback

        traceback.print_exc()


def generate_final_discussion(results_df, detailed_results):
    for task in ["binary", "multiclass"]:
        task_name = "Progressão CKD" if task == "binary" else "Estágio CKD"
        task_results = results_df[results_df["Tarefa"] == task]

        if not task_results.empty:
            print(f"\n--- {task_name} ({task}) ---")

            best_accuracy = task_results.loc[task_results["Acurácia"].idxmax()]
            best_f1 = task_results.loc[task_results["F1-Score"].idxmax()]
            best_auc = task_results.loc[task_results["AUC-ROC"].idxmax()]

            print(
                f"• Melhor Acurácia: {best_accuracy['Modelo']} ({best_accuracy['Acurácia']:.4f})"
            )
            print(f"• Melhor F1-Score: {best_f1['Modelo']} ({best_f1['F1-Score']:.4f})")
            print(f"• Melhor AUC-ROC: {best_auc['Modelo']} ({best_auc['AUC-ROC']:.4f})")

            task_results["Score_Balanceado"] = (
                task_results["Acurácia"]
                + task_results["F1-Score"]
                + task_results["AUC-ROC"]
            ) / 3
            best_balanced = task_results.loc[task_results["Score_Balanceado"].idxmax()]
            print(f"• Modelo mais equilibrado: {best_balanced['Modelo']}")

    overall_best = results_df.groupby("Modelo")["F1-Score"].mean().idxmax()
    overall_score = results_df.groupby("Modelo")["F1-Score"].mean().max()

    print(
        f"• Modelo recomendado geral: {overall_best} (F1-Score médio: {overall_score:.4f})"
    )


if __name__ == "__main__":
    main()
