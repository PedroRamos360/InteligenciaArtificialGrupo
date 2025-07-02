"""
Módulimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import warnings
import os

warnings.filterwarnings('ignore')processamento para Classificação CKD

Este módulo implementa todas as etapas de pré-processamento necessárias:
- Tratamento de valores faltantes
- Codificação de variáveis categóricas
- Normalização/padronização
- Divisão dos dados em treino/validação/teste
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.filterwarnings("ignore")


class CKDPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []

    def load_data(self, data_path):
        self.df = pd.read_csv(data_path)
        print(
            f"Dados carregados: {self.df.shape[0]} registros, {self.df.shape[1]} colunas"
        )
        return self.df

    def handle_missing_values(self, df):
        df_processed = df.copy()
        print("\n=== TRATAMENTO DE VALORES FALTANTES ===")
        missing_cols = df_processed.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        if len(missing_cols) > 0:
            print("Colunas com valores faltantes:")
            for col, count in missing_cols.items():
                print(f"  {col}: {count} valores ({count/len(df)*100:.1f}%)")
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy="median")
                    df_processed[col] = (
                        self.imputers[col].fit_transform(df_processed[[col]]).flatten()
                    )
                else:
                    df_processed[col] = (
                        self.imputers[col].transform(df_processed[[col]]).flatten()
                    )
                print(f"  {col}: preenchido com mediana")
        categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy="most_frequent")
                    df_processed[col] = (
                        self.imputers[col].fit_transform(df_processed[[col]]).flatten()
                    )
                else:
                    df_processed[col] = (
                        self.imputers[col].transform(df_processed[[col]]).flatten()
                    )
                print(f"  {col}: preenchido com moda")
        print(f"Valores faltantes após tratamento: {df_processed.isnull().sum().sum()}")
        return df_processed

    def encode_categorical_variables(self, df):
        df_encoded = df.copy()
        print("\n=== CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS ===")
        already_encoded = [
            "Sex",
            "CKD_Cause",
            "CKD_Stage",
            "CKD_Risk",
            "Dipstick_Proteinuria",
            "Proteinuria",
            "Occult_Blood_in_Urine",
            "UPCR_Severity",
            "Hypertension",
            "Previous_CVD",
            "Diabetes",
            "RAAS_Inhibitor",
            "Calcium_Channel_Blocker",
            "Diuretics",
            "CKD_Progression",
        ]
        categorical_cols = df_encoded.select_dtypes(exclude=[np.number]).columns
        categorical_cols = [
            col for col in categorical_cols if col not in already_encoded
        ]
        if len(categorical_cols) > 0:
            print("Codificando variáveis categóricas:")
            for col in categorical_cols:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col])
                else:
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col])
                print(f"  {col}: {len(self.encoders[col].classes_)} classes únicas")
        else:
            print("Todas as variáveis categóricas já estão codificadas numericamente")
        return df_encoded

    def prepare_features_and_targets(self, df):
        target_cols = ["CKD_Stage", "CKD_Progression"]
        leakage_cols = [
            "eGFR",
            "Creatinine",
            "CKD_Risk",
        ]
        exclude_cols = target_cols + leakage_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y_stage = df["CKD_Stage"].copy()
        y_progression = df["CKD_Progression"].copy()
        self.feature_names = feature_cols
        print(f"\n=== PREPARAÇÃO DAS VARIÁVEIS ===")
        print(f"Features: {len(feature_cols)} variáveis")
        print(f"Variáveis removidas (data leakage): {leakage_cols}")
        print(f"Target 1 (CKD_Stage): {len(y_stage.unique())} classes")
        print(f"Target 2 (CKD_Progression): {len(y_progression.unique())} classes")

        return X, y_stage, y_progression

    def normalize_features(self, X_train, X_val=None, X_test=None):
        print("\n=== NORMALIZAÇÃO DAS FEATURES ===")
        if "features" not in self.scalers:
            self.scalers["features"] = StandardScaler()
            X_train_scaled = self.scalers["features"].fit_transform(X_train)
        else:
            X_train_scaled = self.scalers["features"].transform(X_train)
        print("Features normalizadas usando StandardScaler")
        print(f"Média após normalização: {X_train_scaled.mean():.4f}")
        print(f"Desvio padrão após normalização: {X_train_scaled.std():.4f}")
        results = [X_train_scaled]
        if X_val is not None:
            X_val_scaled = self.scalers["features"].transform(X_val)
            results.append(X_val_scaled)
        if X_test is not None:
            X_test_scaled = self.scalers["features"].transform(X_test)
            results.append(X_test_scaled)
        return tuple(results) if len(results) > 1 else results[0]

    def split_data(
        self, X, y_stage, y_progression, test_size=0.15, val_size=0.15, random_state=42
    ):
        print(f"\n=== DIVISÃO DOS DADOS ===")
        print(f"Treino: {(1-test_size-val_size)*100:.0f}%")
        print(f"Validação: {val_size*100:.0f}%")
        print(f"Teste: {test_size*100:.0f}%")
        X_temp, X_test, y_stage_temp, y_stage_test, y_prog_temp, y_prog_test = (
            train_test_split(
                X,
                y_stage,
                y_progression,
                test_size=test_size,
                random_state=random_state,
                stratify=y_progression,
            )
        )
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_stage_train, y_stage_val, y_prog_train, y_prog_val = (
            train_test_split(
                X_temp,
                y_stage_temp,
                y_prog_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=y_prog_temp,
            )
        )
        print(f"Tamanhos finais:")
        print(f"  Treino: {len(X_train)} registros")
        print(f"  Validação: {len(X_val)} registros")
        print(f"  Teste: {len(X_test)} registros")
        return (
            X_train,
            X_val,
            X_test,
            y_stage_train,
            y_stage_val,
            y_stage_test,
            y_prog_train,
            y_prog_val,
            y_prog_test,
        )

    def save_preprocessors(self, save_path):
        preprocessors = {
            "scalers": self.scalers,
            "encoders": self.encoders,
            "imputers": self.imputers,
            "feature_names": self.feature_names,
        }
        joblib.dump(preprocessors, f"{save_path}/preprocessors.pkl")
        print(f"Preprocessadores salvos em {save_path}/preprocessors.pkl")

    def load_preprocessors(self, load_path):
        preprocessors = joblib.load(f"{load_path}/preprocessors.pkl")
        self.scalers = preprocessors["scalers"]
        self.encoders = preprocessors["encoders"]
        self.imputers = preprocessors["imputers"]
        self.feature_names = preprocessors["feature_names"]
        print(f"Preprocessadores carregados de {load_path}/preprocessors.pkl")

    def full_preprocessing_pipeline(self, data_path, save_path=None):
        print("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO...")
        print("=" * 60)
        df = self.load_data(data_path)
        df_clean = self.handle_missing_values(df)
        df_encoded = self.encode_categorical_variables(df_clean)
        X, y_stage, y_progression = self.prepare_features_and_targets(df_encoded)
        data_splits = self.split_data(X, y_stage, y_progression)
        (
            X_train,
            X_val,
            X_test,
            y_stage_train,
            y_stage_val,
            y_stage_test,
            y_prog_train,
            y_prog_val,
            y_prog_test,
        ) = data_splits
        X_train_scaled, X_val_scaled, X_test_scaled = self.normalize_features(
            X_train, X_val, X_test
        )
        if save_path:
            self.save_preprocessors(save_path)
        print("\nPré-processamento concluído com sucesso!")
        return {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_stage_train": y_stage_train,
            "y_stage_val": y_stage_val,
            "y_stage_test": y_stage_test,
            "y_prog_train": y_prog_train,
            "y_prog_val": y_prog_val,
            "y_prog_test": y_prog_test,
            "feature_names": self.feature_names,
        }


if __name__ == "__main__":
    data_path = "ckd.csv"
    save_path = "."
    preprocessor = CKDPreprocessor()
    processed_data = preprocessor.full_preprocessing_pipeline(data_path, save_path)
    print(
        f"\nDados processados salvos. Shape das features de treino: {processed_data['X_train'].shape}"
    )
