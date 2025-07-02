# ARQUIVOS GERADOS PELO PROJETO CKD

Este documento lista todos os arquivos gerados durante a execução do projeto de classificação de doenças renais crônicas.

## 📊 GRÁFICOS E VISUALIZAÇÕES (pasta `graphs/`)

### Análise Exploratória de Dados

- **`target_distribution.png`** - Distribuição das variáveis alvo (CKD_Stage e CKD_Progression)
- **`numeric_distributions.png`** - Histogramas das principais variáveis numéricas
- **`correlation_matrix.png`** - Matriz de correlação entre variáveis numéricas
- **`boxplots_by_stage.png`** - Boxplots das variáveis por estágio CKD
- **`progression_analysis.png`** - Análise da progressão por características demográficas

### Matrizes de Confusão - Teste Simplificado

- **`confusion_simple_Decision_Tree_binary.png`** - Decision Tree para progressão
- **`confusion_simple_MLP_binary.png`** - MLP para progressão
- **`confusion_simple_Random_Forest_binary.png`** - Random Forest para progressão
- **`confusion_simple_Decision_Tree_multiclass.png`** - Decision Tree para estágios
- **`confusion_simple_MLP_multiclass.png`** - MLP para estágios
- **`confusion_simple_Random_Forest_multiclass.png`** - Random Forest para estágios

### Matrizes de Confusão - Pipeline Completo (quando executado)

- **`confusion_matrix_Decision_Tree_binary.png`** - Decision Tree otimizado
- **`confusion_matrix_MLP_binary.png`** - MLP otimizado
- **`confusion_matrix_Random_Forest_binary.png`** - Random Forest otimizado
- **`confusion_matrix_Decision_Tree_multiclass.png`** - Decision Tree para estágios
- **`confusion_matrix_MLP_multiclass.png`** - MLP para estágios
- **`confusion_matrix_Random_Forest_multiclass.png`** - Random Forest para estágios

### Curvas ROC (quando pipeline completo é executado)

- **`roc_curves_binary.png`** - Curvas ROC para classificação binária
- **`roc_curves_multiclass.png`** - Curvas ROC para classificação multiclasse
- **`model_comparison.png`** - Gráfico comparativo dos modelos

## 📋 DADOS E RESULTADOS

### Tabelas de Resultados

- **`model_comparison.csv`** - Tabela comparativa detalhada dos modelos
- **`simple_model_comparison.csv`** - Resultados do teste simplificado

### Modelos Treinados

- **`trained_models.pkl`** - Modelos otimizados (pipeline completo)
- **`simple_trained_models.pkl`** - Modelos do teste simplificado
- **`preprocessors.pkl`** - Objetos de pré-processamento (scaler, encoder, etc.)

### Resultados Detalhados

- **`model_results.pkl`** - Resultados completos com métricas detalhadas
- **`simple_model_results.pkl`** - Resultados do teste simplificado
- **`best_parameters.pkl`** - Melhores hiperparâmetros encontrados

## 📁 ESTRUTURA DE ARQUIVOS

```
t2/
├── ckd.csv                              # Dataset original
├── requirements.txt                     # Dependências Python
├── README.md                           # Documentação do projeto
├── RELATORIO_FINAL.md                  # Relatório acadêmico completo
├── ARQUIVOS_GERADOS.md                 # Este arquivo
│
├── 📁 scripts/
│   ├── main.py                         # Pipeline completo
│   ├── simple_test.py                  # Teste simplificado
│   ├── demo_usage.py                   # Demonstração prática
│   ├── exploratory_analysis.py         # Análise exploratória
│   ├── preprocessing.py                # Pré-processamento
│   ├── models.py                       # Modelos de classificação
│   └── install_dependencies.py         # Instalador de dependências
│
├── 📁 graphs/                          # GRÁFICOS GERADOS
│   ├── target_distribution.png
│   ├── numeric_distributions.png
│   ├── correlation_matrix.png
│   ├── boxplots_by_stage.png
│   ├── progression_analysis.png
│   ├── confusion_simple_*.png
│   ├── confusion_matrix_*.png
│   ├── roc_curves_*.png
│   └── model_comparison.png
│
├── 📁 dados_gerados/
│   ├── *.csv                          # Tabelas de resultados
│   ├── *.pkl                          # Modelos e preprocessadores
│   └── *.json                         # Configurações (se houver)
│
└── __pycache__/                        # Cache Python
```

## 🔧 COMO GERAR OS ARQUIVOS

### Teste Rápido (Recomendado)

```bash
python simple_test.py
```

**Gera:** Gráficos básicos, modelos simples, resultados essenciais

### Pipeline Completo

```bash
python main.py
```

**Gera:** Todos os arquivos, otimização completa, relatórios detalhados

### Análise Individual

```bash
python exploratory_analysis.py    # Apenas gráficos exploratórios
python preprocessing.py           # Apenas pré-processamento
```

## 📝 NOTAS IMPORTANTES

1. **Pasta `graphs/`**: Criada automaticamente pelos scripts
2. **Arquivos `.pkl`**: Requerem Python para serem lidos
3. **Imagens PNG**: Podem ser visualizadas diretamente
4. **Tabelas CSV**: Podem ser abertas no Excel ou editores de texto

## 🔍 VERIFICAÇÃO DOS ARQUIVOS

Para verificar se todos os arquivos foram gerados corretamente:

```bash
# Listar arquivos da pasta graphs
ls graphs/

# Verificar modelos salvos
python -c "import joblib; print(list(joblib.load('simple_trained_models.pkl').keys()))"
```

---

_Documento atualizado automaticamente durante a execução do projeto._
