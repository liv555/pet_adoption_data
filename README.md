# 🐾 Pipeline Completo de Machine Learning — Adoção de Pets

> Sistema de Predição com ML Integrado (PyCaret + Scikit-Learn)

---

## 📋 Descrição

Este projeto implementa um **pipeline completo de Machine Learning** aplicado ao problema de adoção de animais em abrigos. Utilizando o dataset **Pet Adoption Data**, o notebook cobre quatro tipos distintos de problemas de ML, desde classificação supervisionada até sistemas de recomendação.

---

## 🎯 Objetivos

| Pipeline | Problema | Objetivo |
|---|---|---|
| 1 — Classificação | Supervisionado (binário) | Prever se um animal **será adotado** (0 ou 1) |
| 2 — Regressão | Supervisionado (contínuo) | Prever **quantos dias** o animal ficará no abrigo |
| 3 — Clusterização | Não-supervisionado | **Segmentar animais** por perfis de comportamento |
| 4 — Recomendação | Item-Item Filtering | **Recomendar animais similares** com base em características |

---

## 📊 Dataset

**Nome:** Pet Adoption Data  
**Fonte:** [GitHub — liv555/pet_adoption_data](https://raw.githubusercontent.com/liv555/pet_adoption_data/main/pet_adoption_data.csv)

### Principais colunas

| Coluna | Tipo | Descrição |
|---|---|---|
| `PetID` | int | Identificador único do animal |
| `PetType` | categórico | Tipo do animal (cão, gato, etc.) |
| `Breed` | categórico | Raça do animal |
| `Size` | categórico | Porte (pequeno, médio, grande) |
| `AgeMonths` | numérico | Idade em meses |
| `WeightKg` | numérico | Peso em quilogramas |
| `Vaccinated` | binário | Se está vacinado (0/1) |
| `HealthCondition` | numérico | Condição de saúde (0 = saudável) |
| `AdoptionFee` | numérico | Taxa de adoção ($) |
| `TimeInShelterDays` | numérico | Dias de permanência no abrigo |
| `PreviousOwner` | binário | Se tinha dono anterior (0/1) |
| `AdoptionLikelihood` | binário | **Variável alvo** — foi adotado? (0/1) |

---

## ⚙️ Requisitos

> **⚠️ Importante:** Este notebook requer **Python ≤ 3.10** devido à compatibilidade com o PyCaret. Utilize um ambiente virtual (venv) para isolar as dependências.

### Dependências principais

```txt
pycaret>=3.0
scikit-learn
pandas
numpy
matplotlib
seaborn
fastapi
joblib
```

### Instalação

```bash
# Criar ambiente virtual com Python 3.10
py -3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependências
py -3.10 -m pip install pycaret scikit-learn pandas numpy matplotlib seaborn
```

---

## 🗂️ Estrutura do Notebook

### 1. Instalação e Configuração
Instalação das dependências e configuração dos imports e parâmetros globais de visualização.

### 2. Carregamento e Validação do Dataset
- Ingestão dos dados via URL pública (com fallback local)
- Verificação de integridade: tipos, nulos, duplicatas e unicidade do `PetID`
- Remoção de registros duplicados
- Estatísticas descritivas (numéricas e categóricas)

### 3. Análise Exploratória de Dados (EDA)
- **Limpeza inicial:** remoção de nulos em colunas essenciais, padronização de textos e filtragem de valores implausíveis
- **Histogramas:** distribuições de `AgeMonths`, `WeightKg` e `AdoptionFee`
- **Boxplots:** detecção de outliers nas variáveis numéricas
- **Engenharia de Features:** criação de `WeightToAgeRatio`, `IsSenior`, `ReadyForAdoption` e agregações por tipo de animal
- **Heatmap de correlação:** análise de Pearson entre todas as variáveis numéricas
- **Feature Importance:** Random Forest com 200 estimadores para identificar as variáveis mais relevantes

### 4. Pré-processamento e Preparação dos Dados
- Capping de outliers no percentil 99
- Agrupamento de raças raras (< 5 ocorrências → `"Other"`)
- Criação de 3 datasets específicos para cada pipeline:
  - `df_classification` — alvo: `AdoptionLikelihood`
  - `df_regression` — alvo: `LogTimeInShelter` (log-transform)
  - `df_clustering` — sem variável alvo

### 5. Pipeline 1 — Classificação (PyCaret)
- Setup com normalização Z-Score, imputação por mediana/moda e balanceamento de classes
- `compare_models()` para seleção automática do melhor algoritmo (métrica: AUC)
- `tune_model()` com 50 iterações de otimização de hiperparâmetros
- Avaliação: Matriz de Confusão, Curva ROC, Curva Precision-Recall, Feature Importance
- Métricas reportadas: Acurácia, Precisão, Recall, F1-Score, AUC-ROC

### 6. Pipeline 2 — Regressão (PyCaret)
- Setup com log-transform no target (`np.log1p(TimeInShelterDays)`)
- `compare_models()` com métrica R²
- `tune_model()` com 50 iterações
- Avaliação: Gráfico de Resíduos, Predito vs Real
- Métricas reportadas: MAE, MSE, RMSE, R² (com conversão de volta para dias reais)

### 7. Pipeline 3 — Clusterização (PyCaret + KMeans)
- Teste de K = 2 a 8 com Silhouette Score para seleção do número ótimo de clusters
- Treinamento do modelo KMeans final com o K ótimo
- Visualizações: PCA 2D, distribuição de clusters, boxplots comparativos por perfil
- Interpretação automática dos clusters (filhotes, sêniores, custo elevado, etc.)

### 8. Pipeline 4 — Sistema de Recomendação (Scikit-Learn)
- Matriz de features com One-Hot Encoding + StandardScaler
- Similaridade de Cosseno Pet-Pet
- Função `recommend_similar_pets(pet_idx, n)` para buscar os N animais mais similares
- Avaliação com **Precision @ K** (K = 5, 10, 20) baseada em correspondência de tipo animal

### 9. Persistência dos Modelos
Todos os artefatos são salvos no diretório `models/`:

| Arquivo | Conteúdo |
|---|---|
| `classification_pipeline.pkl` | Pipeline de classificação (PyCaret) |
| `clustering_pipeline.pkl` | Modelo KMeans (PyCaret) |
| `pet_similarity.pkl` | Matriz de similaridade de cosseno |
| `rec_scaler.pkl` | StandardScaler do sistema de recomendação |
| `df_rec.pkl` | DataFrame de referência para recomendação |
| `features_info.pkl` | Dicionário com listas de features por pipeline |

---

## 📈 Métricas e Avaliação

| Pipeline | Métricas |
|---|---|
| Classificação | Acurácia, Precisão, Recall, F1-Score, AUC-ROC, Matriz de Confusão |
| Regressão | MAE, MSE, RMSE, R² |
| Clusterização | Silhouette Score, Calinski-Harabasz, Davies-Bouldin |
| Recomendação | Precision @ K (K = 5, 10, 20) |

---

## 🖼️ Arquivos Gerados

| Arquivo | Descrição |
|---|---|
| `eda_pets_histogramas.png` | Distribuições de Idade, Peso e Taxa de Adoção |
| `eda_pets_boxplots.png` | Boxplots para detecção de outliers |
| `eda_pets_correlacao.png` | Heatmap de correlação de Pearson |
| `eda_pets_importancia.png` | Feature Importance (Random Forest) |
| `eda_pets_complementar.png` | Análise complementar (tipos, tempo, dispersão) |
| `clustering_silhouette.png` | Seleção do número ótimo de clusters |
| `clustering_perfis.png` | Boxplots comparativos por cluster |
| `recomendacao_similaridade.png` | Heatmap de similaridade (top 20 pets) |
| `recomendacao_precision.png` | Precision @ K do sistema de recomendação |

---

## 🚀 Como Executar

1. Clone ou baixe o repositório
2. Certifique-se de usar **Python ≤ 3.10**
3. Instale as dependências (seção de Requisitos)
4. Execute o notebook célula por célula, na ordem indicada pelo sumário
5. Os modelos serão salvos automaticamente na pasta `models/`

> O dataset é carregado automaticamente via URL. Se não houver conexão, coloque o arquivo `pet_adoption_data.csv` na mesma pasta do notebook.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10**
- **PyCaret 3.x** — AutoML para classificação, regressão e clusterização
- **Scikit-Learn** — Feature Importance, Similaridade de Cosseno, métricas
- **Pandas / NumPy** — Manipulação e transformação de dados
- **Matplotlib / Seaborn** — Visualizações

---

## ✅ Requisitos do TCC Atendidos

- ✅ Coleta e carregamento de dados externos
- ✅ Validação de integridade do dataset
- ✅ Tratamento de variáveis (imputação, codificação, normalização)
- ✅ Análise Exploratória de Dados (EDA) com visualizações
- ✅ Engenharia de Features
- ✅ Pipeline de Classificação supervisionada
- ✅ Pipeline de Regressão supervisionada
- ✅ Pipeline de Clusterização não-supervisionada
- ✅ Sistema de Recomendação
- ✅ Treinamento, tuning e avaliação dos modelos
- ✅ Persistência dos modelos treinados
