# Projeto: Previsão de Renda

Este repositório contém a análise exploratória e a modelagem preditiva para o projeto de Previsão de Renda, desenvolvido como parte do curso de Ciência de Dados da EBAC.

## 1. Coleta de Dados
Os dados foram fornecidos em formato CSV (`previsao_de_renda_II.csv`), contendo 750.000 registros com informações socioeconômicas, tais como:
- **Renda**: Variável alvo.
- **Tempo de Emprego**: Variável explicativa chave.
- **Sexo, Idade, Escolaridade**: Variáveis demográficas.
- **Posses (Veículo/Imóvel)**: Indicadores patrimoniais.

## 2. Modelagem
O projeto seguiu um pipeline robusto de Ciência de Dados:
- **Pré-processamento**: Tratamento de valores nuloes (mediana para tempo de emprego) e conversão de colunas temporais.
- **Tratamento de Outliers**: Aplicação de *capping* no percentil 99 para mitigar o impacto de rendas extremas no modelo.
- **Engenharia de Variáveis**: Transformação de variáveis categóricas via *One-Hot Encoding*.
- **Algoritmo**: Utilização do `RandomForestRegressor`, um modelo de ensamble potente para capturar relações não lineares.
- **Otimização**: O código está estruturado para permitir buscas de hiperparâmetros (RandomizedSearchCV).

## 3. Conclusões
- O modelo de **Random Forest** demonstrou alta capacidade de explicar a variabilidade da renda.
- A variável **Tempo de Emprego** confirmou-se como um dos principais preditores de renda.
- O tratamento de **outliers** via capping mostrou-se essencial para estabilizar os erros de previsão (MSE) e melhorar a generalização do modelo.
- A visualização dos dados aponta correlações claras entre estabilidade profissional e incremento nos vencimentos.

---
*Desenvolvido por Arthur.*
