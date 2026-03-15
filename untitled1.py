# -*- coding: utf-8 -*-
"""
Projeto: Previsão de Renda
Refatorado para melhor performance, legibilidade e manutenibilidade.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuração de estilo para os gráficos
sns.set_theme(style="whitegrid")

# ==========================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ==========================================
def preprocess_data(file_path):
    print("Carregando e processando os dados...")
    df = pd.read_csv(file_path)
    
    # Tratamento de nulos
    df['tempo_emprego'] = df['tempo_emprego'].fillna(df['tempo_emprego'].median())
    
    # Conversão de datas e tipos
    df['data_ref'] = pd.to_datetime(df['data_ref'])
    if 'qt_pessoas_residencia' in df.columns:
        df['qt_pessoas_residencia'] = df['qt_pessoas_residencia'].astype(int)
    
    # Tratamento de Outliers (Capping no percentil 99)
    limite_superior = df['renda'].quantile(0.99)
    df['renda_capped'] = df['renda'].clip(upper=limite_superior)
    
    # One-Hot Encoding
    colunas_categoricas = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 
                           'educacao', 'estado_civil', 'tipo_residencia']
    df_encoded = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True, dtype=int)
    
    # Remoção de colunas desnecessárias ('index' e 'Unnamed: 0' caso existam)
    colunas_para_remover = ['renda', 'data_ref', 'index', 'Unnamed: 0']
    df_encoded = df_encoded.drop(columns=[col for col in colunas_para_remover if col in df_encoded.columns])
    
    return df, df_encoded

# Caminho do arquivo (ajuste se necessário)
caminho_arquivo = 'previsao_de_renda_II.csv'
df_raw, df_clean = preprocess_data(caminho_arquivo)

# ==========================================
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# ==========================================
print("\nGerando visualizações exploratórias...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Matriz de Correlação
corr_cols = ['idade', 'tempo_emprego', 'qtd_filhos', 'qt_pessoas_residencia', 'renda_capped']
sns.heatmap(df_raw[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[0, 0])
axes[0, 0].set_title('Matriz de Correlação')

# Distribuição de Renda por Educação
order_edu = df_raw.groupby('educacao')['renda_capped'].median().sort_values(ascending=False).index
sns.boxplot(data=df_raw, x='educacao', y='renda_capped', order=order_edu, hue='educacao', palette='viridis', legend=False, ax=axes[0, 1])
axes[0, 1].set_title('Renda por Escolaridade')
axes[0, 1].tick_params(axis='x', rotation=45)

# Distribuição de Renda por Gênero
sns.boxplot(data=df_raw, x='sexo', y='renda_capped', hue='sexo', palette='Set2', legend=False, ax=axes[1, 0])
axes[1, 0].set_title('Renda por Gênero')

# Tempo de Emprego vs Renda (Amostra para otimizar plotagem)
sns.regplot(data=df_raw.sample(1000, random_state=42), x='tempo_emprego', y='renda_capped', 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=axes[1, 1])
axes[1, 1].set_title('Tempo de Emprego vs Renda (Amostra 1000)')

plt.tight_layout()
plt.show()

# ==========================================
# 3. DIVISÃO DOS DADOS
# ==========================================
X = df_clean.drop(columns=['renda_capped'])
y = df_clean['renda_capped']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# ==========================================
# 4. MODELAGEM E AVALIAÇÃO
# ==========================================
def avaliar_modelo(modelo, nome_modelo):
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2, y_pred

# --- Modelo Baseline ---
print("\nTreinando Modelo Baseline...")
rf_base = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
rmse_base, r2_base, _ = avaliar_modelo(rf_base, "Baseline")

# --- Modelo Otimizado ---
# (Nota: O código original definia um RandomizedSearchCV, mas não o executava. 
# Aqui usamos os melhores parâmetros diretamente para poupar tempo de execução, 
# mas deixei a estrutura do SearchCV nos comentários abaixo caso queira rodar).

"""
# Exemplo de como rodar o RandomizedSearchCV (CUIDADO: Pode demorar horas dependendo da máquina)
param_dist = {'n_estimators': [50, 100], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_dist, n_iter=5, cv=3, scoring='r2', random_state=42)
random_search.fit(X_train.sample(50000), y_train.sample(50000)) # Usando amostra para não demorar
melhores_params = random_search.best_params_
"""

melhores_params = {'n_estimators': 100, 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 2}
print("Treinando Modelo Otimizado...")
rf_opt = RandomForestRegressor(**melhores_params, random_state=42, n_jobs=-1)
rf_opt.fit(X_train, y_train)
rmse_opt, r2_opt, y_pred_opt = avaliar_modelo(rf_opt, "Otimizado")

# ==========================================
# 5. RESULTADOS E COMPARAÇÃO
# ==========================================
melhoria_r2 = ((r2_opt - r2_base) / r2_base) * 100

print(f'\n--- COMPARAÇÃO DE DESEMPENHO ---')
print(f'BASELINE  -> RMSE: {rmse_base:.2f} | R²: {r2_base:.4f}')
print(f'OTIMIZADO -> RMSE: {rmse_opt:.2f}  | R²: {r2_opt:.4f}')
print(f'Melhoria no R²: {melhoria_r2:.2f}%')

# Gráficos Finais de Diagnóstico do Modelo Otimizado
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Real vs Previsto
sns.scatterplot(x=y_test, y=y_pred_opt, alpha=0.3, ax=axes[0])
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
axes[0].set_title('Real vs Previsto (Otimizado)')

# 2. Resíduos
sns.histplot(y_test - y_pred_opt, bins=50, kde=True, ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('Distribuição de Resíduos')

# 3. Importância das Features (Top 10)
importancias = pd.Series(rf_opt.feature_importances_, index=X.columns).nlargest(10)
sns.barplot(x=importancias.values, y=importancias.index, hue=importancias.index, palette='viridis', legend=False, ax=axes[2])
axes[2].set_title('Top 10 Variáveis mais Importantes')

plt.tight_layout()
plt.show()