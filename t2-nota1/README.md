# Trabalho de Classificação de Doenças Renais Crônicas (CKD)

## 📋 Descrição do Projeto

Este projeto implementa técnicas de aprendizado de máquina para classificação de doenças renais crônicas (CKD), abordando dois problemas principais:

1. **CKD_Stage**: Classificação multiclasse (estágios 1-5 da DRC)
2. **CKD_Progression**: Classificação binária (progressão ou não da DRC)

## 🗂️ Estrutura do Projeto

```
t2/
├── ckd.csv                     # Dataset original
├── requirements.txt            # Dependências Python
├── main.py                     # Script principal
├── simple_test.py              # Teste simplificado (mais rápido)
├── demo_usage.py               # Demonstração prática dos modelos
├── exploratory_analysis.py     # Análise exploratória de dados
├── preprocessing.py            # Pré-processamento dos dados
├── models.py                   # Modelos de classificação
├── README.md                   # Este arquivo
├── RELATORIO_FINAL.md          # Relatório acadêmico completo
├── ARQUIVOS_GERADOS.md         # Lista de arquivos gerados
│
├── graphs/                     # 📊 Gráficos gerados automaticamente
│   ├── target_distribution.png        # Distribuição das variáveis alvo
│   ├── numeric_distributions.png      # Histogramas das variáveis
│   ├── correlation_matrix.png         # Matriz de correlação
│   ├── boxplots_by_stage.png          # Boxplots por estágio CKD
│   ├── progression_analysis.png       # Análise de progressão
│   └── confusion_*.png                # Matrizes de confusão dos modelos
│
└── [outros arquivos gerados]   # Modelos (.pkl), resultados (.csv)
```

## 🚀 Como Executar

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o Pipeline Completo

```bash
# Pipeline completo (pode demorar mais)
python main.py

# OU teste simplificado (mais rápido, recomendado)
python simple_test.py
```

### 3. Demonstração Prática

```bash
# Ver como usar os modelos treinados
python demo_usage.py
```

### 4. Executar Módulos Individualmente (Opcional)

```bash
# Apenas análise exploratória
python exploratory_analysis.py

# Apenas pré-processamento
python preprocessing.py
```

## 📊 Modelos Implementados

### 1. Árvores de Decisão (Decision Tree)

- **Vantagens**: Interpretabilidade, não requer normalização
- **Hiperparâmetros**: max_depth, min_samples_split, min_samples_leaf, criterion

### 2. Redes Neurais Artificiais (MLP)

- **Vantagens**: Capacidade de capturar relações não-lineares complexas
- **Hiperparâmetros**: hidden_layer_sizes, activation, alpha, learning_rate

### 3. Random Forest (Técnica Adicional Escolhida)

- **Vantagens**: Robustez, redução de overfitting, importância das features
- **Hiperparâmetros**: n_estimators, max_depth, min_samples_split, min_samples_leaf

## 🔍 Metodologia

### Análise Exploratória

- Estatísticas descritivas das variáveis
- Análise de valores faltantes
- Visualizações (histogramas, boxplots, correlações)
- Análise das variáveis alvo

### Pré-processamento

- Tratamento de valores faltantes (mediana para numéricas, moda para categóricas)
- Codificação de variáveis categóricas
- Normalização com StandardScaler
- Divisão: 70% treino, 15% validação, 15% teste

### Avaliação

- **Métricas**: Acurácia, Precisão, Recall, F1-Score, AUC-ROC
- **Visualizações**: Matriz de confusão, Curvas ROC
- **Otimização**: Grid Search com validação cruzada

## 📈 Arquivos Gerados

### 📊 Gráficos (pasta `graphs/`)

**Análise Exploratória:**

- `target_distribution.png` - Distribuição das variáveis alvo
- `numeric_distributions.png` - Histogramas das variáveis numéricas
- `correlation_matrix.png` - Matriz de correlação
- `boxplots_by_stage.png` - Boxplots por estágio CKD
- `progression_analysis.png` - Análise de progressão

**Resultados dos Modelos:**

- `confusion_simple_*.png` - Matrizes de confusão (teste simplificado)
- `confusion_matrix_*.png` - Matrizes de confusão (pipeline completo)
- `roc_curves_*.png` - Curvas ROC comparativas
- `model_comparison.png` - Gráfico comparativo dos modelos
- `roc_curves_*.png` - Curvas ROC
- `model_comparison.png` - Comparação dos modelos

### Dados e Modelos

- `model_comparison.csv` - Tabela comparativa dos resultados
- `trained_models.pkl` - Modelos treinados
- `model_results.pkl` - Resultados detalhados
- `best_parameters.pkl` - Melhores hiperparâmetros
- `preprocessors.pkl` - Objetos de pré-processamento

## 🏥 Contexto Clínico

### Importância da DRC

- Doença silenciosa que afeta milhões mundialmente
- Diagnóstico precoce é crucial para prevenção de complicações
- Estágios baseados na taxa de filtração glomerular (eGFR)

### Implicações dos Resultados

- **Falsos Positivos**: Tratamentos desnecessários, custos adicionais
- **Falsos Negativos**: CRÍTICO - atraso no tratamento, progressão da doença
- **Prioridade**: Minimizar falsos negativos (alta sensibilidade)

## 📋 Features do Dataset

### Variáveis Demográficas

- **Sex**: Gênero (1=Masculino, 2=Feminino)
- **Age**: Idade em anos

### Variáveis Clínicas

- **Systolic_Pressure**: Pressão arterial sistólica (mmHg)
- **BMI**: Índice de Massa Corporal
- **Hemoglobin**: Hemoglobina (g/dL)
- **Albumin**: Albumina sérica (g/dL)
- **Creatinine**: Creatinina sérica (mg/dL)
- **eGFR**: Taxa de filtração glomerular (mL/min/1,73m²)

### Variáveis de Comorbidades

- **Hypertension**: Hipertensão (0=Não, 1=Sim)
- **Previous_CVD**: Histórico cardiovascular
- **Diabetes**: Diabetes
- **CKD_Cause**: Causa da DRC

### Variáveis Laboratoriais

- **Dipstick_Proteinuria**: Resultado proteinúria
- **Proteinuria**: Presença de proteinúria
- **Occult_Blood_in_Urine**: Sangue oculto na urina
- **Protein_Creatinine_Ratio**: Relação proteína/creatinina
- **UPCR_Severity**: Gravidade da UPCR

### Medicações

- **RAAS_Inhibitor**: Inibidores da RAA
- **Calcium_Channel_Blocker**: Bloqueadores de cálcio
- **Diuretics**: Diuréticos

### Variáveis Alvo

- **CKD_Stage**: Estágio da DRC (1-5) - MULTICLASSE
- **CKD_Progression**: Progressão (0=Não, 1=Sim) - BINÁRIA

## 🎯 Resultados Esperados

### Métricas de Performance

- Acurácia > 80% para ambas as tarefas
- F1-Score balanceado entre precisão e recall
- AUC-ROC > 0.85 para boa discriminação

### Insights Clínicos

- Identificação das variáveis mais importantes
- Padrões de progressão da doença
- Fatores de risco modificáveis

## ⚠️ Limitações

- Tamanho limitado do dataset
- Possível desbalanceamento entre classes
- Necessidade de validação externa
- Interpretabilidade limitada (especialmente MLP)

## 🔮 Melhorias Futuras

- Técnicas de balanceamento (SMOTE)
- Feature engineering avançada
- Ensemble methods
- Validação temporal
- Análise de importância das features
- Calibração de probabilidades

## 👥 Autores

**[INSERIR NOMES DOS ALUNOS AQUI]**

## 📅 Data

Junho 2025

## 📝 Notas para o Relatório

### Estrutura Sugerida do PDF

1. **Introdução**

   - Contexto da DRC
   - Objetivos do trabalho
   - Importância do ML na medicina

2. **Metodologia**

   - Descrição do dataset
   - Análise exploratória
   - Pré-processamento
   - Modelos implementados
   - Métricas de avaliação

3. **Resultados**

   - Tabelas comparativas
   - Gráficos e visualizações
   - Análise estatística

4. **Discussão**

   - Interpretação dos resultados
   - Implicações clínicas
   - Limitações

5. **Conclusões**
   - Modelo recomendado
   - Contribuições do trabalho
   - Trabalhos futuros

### Anexos

- Código-fonte comentado
- Tabelas detalhadas
- Gráficos adicionais
