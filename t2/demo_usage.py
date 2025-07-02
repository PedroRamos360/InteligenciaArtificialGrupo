"""
Script de Demonstração - Uso Prático dos Modelos CKD

Este script demonstra como usar os modelos treinados para fazer
predições em novos pacientes.
"""

import pandas as pd
import numpy as np
import joblib
from preprocessing import CKDPreprocessor


def load_trained_models():
    try:
        models = joblib.load("simple_trained_models.pkl")
        print("Modelos carregados com sucesso!")
        return models
    except FileNotFoundError:
        print("Modelos não encontrados. Execute primeiro o simple_test.py")
        return None


def create_sample_patients():
    patient_1 = {
        "Sex": 1,
        "Age": 75,
        "Systolic_Pressure": 160,
        "BMI": 28.5,
        "CKD_Cause": 1,
        "Hemoglobin": 9.5,
        "Albumin": 3.2,
        "Creatinine": 4.5,
        "eGFR": 15,
        "CKD_Risk": 10,
        "Dipstick_Proteinuria": 3,
        "Proteinuria": 1,
        "Occult_Blood_in_Urine": 1,
        "Protein_Creatinine_Ratio": 2.5,
        "UPCR_Severity": 3,
        "Hypertension": 1,
        "Previous_CVD": 1,
        "Diabetes": 1,
        "RAAS_Inhibitor": 1,
        "Calcium_Channel_Blocker": 1,
        "Diuretics": 1,
    }
    patient_2 = {
        "Sex": 2,
        "Age": 45,
        "Systolic_Pressure": 125,
        "BMI": 22.0,
        "CKD_Cause": 4,
        "Hemoglobin": 13.5,
        "Albumin": 4.2,
        "Creatinine": 1.1,
        "eGFR": 65,
        "CKD_Risk": 3,
        "Dipstick_Proteinuria": 0,
        "Proteinuria": 0,
        "Occult_Blood_in_Urine": 0,
        "Protein_Creatinine_Ratio": 0.2,
        "UPCR_Severity": 1,
        "Hypertension": 0,
        "Previous_CVD": 0,
        "Diabetes": 0,
        "RAAS_Inhibitor": 0,
        "Calcium_Channel_Blocker": 0,
        "Diuretics": 0,
    }
    patient_3 = {
        "Sex": 1,
        "Age": 60,
        "Systolic_Pressure": 140,
        "BMI": 25.5,
        "CKD_Cause": 2,
        "Hemoglobin": 11.8,
        "Albumin": 3.8,
        "Creatinine": 2.1,
        "eGFR": 35,
        "CKD_Risk": 6,
        "Dipstick_Proteinuria": 1,
        "Proteinuria": 1,
        "Occult_Blood_in_Urine": 0,
        "Protein_Creatinine_Ratio": 0.8,
        "UPCR_Severity": 2,
        "Hypertension": 1,
        "Previous_CVD": 0,
        "Diabetes": 0,
        "RAAS_Inhibitor": 1,
        "Calcium_Channel_Blocker": 0,
        "Diuretics": 0,
    }
    patients_df = pd.DataFrame([patient_1, patient_2, patient_3])
    patients_df.index = [
        "Paciente 1 (Alto Risco)",
        "Paciente 2 (Baixo Risco)",
        "Paciente 3 (Risco Moderado)",
    ]
    return patients_df


def predict_patient_outcomes(models, patients_df):
    try:
        preprocessor = CKDPreprocessor()
        preprocessor.load_preprocessors(".")
    except:
        print("Preprocessador não encontrado. Execute primeiro o preprocessing.py")
        return
    patients_processed = patients_df.copy()
    patients_scaled = preprocessor.scalers["features"].transform(patients_processed)
    print("\n" + "=" * 60)
    print("PREDIÇÕES PARA PACIENTES DE EXEMPLO")
    print("=" * 60)
    models_to_test = [
        ("Decision_Tree_binary", "Árvore de Decisão"),
        ("MLP_binary", "Rede Neural (MLP)"),
        ("Random_Forest_binary", "Random Forest"),
    ]
    results = {}
    for model_key, model_name in models_to_test:
        if model_key in models:
            model = models[model_key]
            predictions = model.predict(patients_scaled)
            probabilities = model.predict_proba(patients_scaled)
            print(f"\n--- {model_name} ---")
            for i, (patient_name, pred, prob) in enumerate(
                zip(patients_df.index, predictions, probabilities)
            ):
                risk_score = prob[1] * 100
                status = "PROGRESSÃO" if pred == 1 else "SEM PROGRESSÃO"
                print(f"{patient_name}:")
                print(f"  Predição: {status}")
                print(f"  Risco de Progressão: {risk_score:.1f}%")
                print(f"  Confiança: {max(prob)*100:.1f}%")
                if patient_name not in results:
                    results[patient_name] = {}
                results[patient_name][model_name] = {
                    "prediction": status,
                    "risk_score": risk_score,
                    "confidence": max(prob) * 100,
                }
    print(f"\n{'='*60}")
    print("RESUMO COMPARATIVO DAS PREDIÇÕES")
    print("=" * 60)
    for patient_name in results:
        print(f"\n{patient_name}:")
        for model_name in results[patient_name]:
            data = results[patient_name][model_name]
            print(
                f"  {model_name}: {data['prediction']} (Risco: {data['risk_score']:.1f}%)"
            )
    return results


def main():
    print("DEMONSTRAÇÃO PRÁTICA - MODELOS CKD")
    print("=" * 50)
    models = load_trained_models()
    if models is None:
        return
    print("\nCRIANDO PACIENTES DE EXEMPLO...")
    patients_df = create_sample_patients()
    print(f"\nDados dos pacientes criados:")
    print(f"• {len(patients_df)} pacientes de exemplo")
    print(f"• {len(patients_df.columns)} variáveis por paciente")
    print(f"• Perfis: Alto risco, Baixo risco, Risco moderado")
    predict_patient_outcomes(models, patients_df)


if __name__ == "__main__":
    main()
