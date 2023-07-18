# -*- coding: utf-8 -*-
"""

Original file is located at
    https://colab.research.google.com/drive/1MuIJIvqYsUicQbzPgJvUAiIwaa-kbetj

#**Adicionando as bibliotecas que serão utilizadas**
"""

import pandas as pd
import numpy as np


import statsmodels.api as sm
import statsmodels.graphics.regressionplots as smg
from scipy import stats


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""#**Abrindo o dataset**"""

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_excel('/content/drive/MyDrive/gps_floripa.xlsx')

df.head()

"""#**Analise exploratória dos dados**"""

len(df)
# O df possui um total de 223402 registros.

# Não existem dados faltantes no dataset.
df.info()

# O dataset possui informações de DuraçãoViagem = 0
df.sort_values(by="DuraçãoViagem", ascending=True).head(5)

# Removendo do dataset onde DuraçãoViagem = 0
# Como poderia 14.600 em 1 min ?
# Como poderia 16.560 em 12:36:00	?

df = df[df["HoraIni"] != df["HoraFim"]]
df.sort_values(by="DuraçãoViagem", ascending=True).head(223402)

df['KmPerc'].describe()

# Transformando a DuraçãoViagem em minutos.
df['DuraçãoViagem'] = df['DuraçãoViagem'].apply(lambda x: x.hour * 60 + x.minute + x.second / 60)

# Criando o histograma da coluna 'DuraçãoViagem'.
plt.hist(df['DuraçãoViagem'], bins=20, color='blue', alpha=0.6)

# Personalizando o gráfico com rótulos e título.
plt.xlabel('DuraçãoViagem')
plt.ylabel('Frequência')
plt.title('HISTROGRAMA DA DURAÇÃO DAS VIAGENS, em minutos')

# Mostrando o histograma na tela.
plt.show()

plt.figure(figsize=(9, 6))  # Define o tamanho da figura.

plt.hist(df['DuraçãoViagem'], bins=20, color='skyblue', edgecolor='navy', alpha=0.8)

# Personalizando o gráfico com rótulos e título.
plt.xlabel('Duração das Viagens (minutos)', fontsize=14)
plt.ylabel('Frequência', fontsize=14)
plt.title('Histograma da Duração das Viagens', fontsize=16)

# Adicionando uma grade ao gráfico.
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionando legenda ao eixo x.
plt.xticks(fontsize=13)

# Adicionando legenda ao eixo y.
plt.yticks(fontsize=13)

# Mostrando o histograma na tela.
plt.tight_layout()
plt.show()

# Verificando a quantidade distinta de linha de onibus em florianopolis, depois do tratamento.
len(df['Linha'].unique())

# Criando uma coluna sobre dia da semana no DF
# Objetivo: Investigar se o dia da semana interfere na duração da viagem
df['DiaSemana'] = df['DataIni'].dt.weekday.map({
    0: 'Segunda-feira',
    1: 'Terça-feira',
    2: 'Quarta-feira',
    3: 'Quinta-feira',
    4: 'Sexta-feira',
    5: 'Sábado',
    6: 'Domingo'
})

df.head()

# Não serão utilizadas DataIni, a informação relavante é o dia da semana
# Não será utilizada  DataFim e hora fim, pois se não, já temos a resposta do que queremos responder.
df.drop(columns=['DataIni',"DataFim", "HoraFim"], inplace=True)

df.head()

# Transformando a hora de inicio em minutos
# Objetivo: usar a datainicio nos modelos
df['Minuto_ini'] = df['HoraIni'].apply(lambda x: x.hour * 60 + x.minute)

df.head()

# Extrai as colunas numéricas do DataFrame, exceto a coluna "DuraçãoViagem".
colunas_numericas = df.select_dtypes(include=[float, int]).columns.tolist()

# Plotando os gráficos de dispersão de todas as colunas numéricas em relação à coluna "DuraçãoViagem".
plt.figure(figsize=(5, 5))
for coluna in colunas_numericas:
    sns.scatterplot(data=df, x="DuraçãoViagem", y=coluna, alpha=0.6)
    plt.title(f"Gráfico de Dispersão: {coluna} em relação a DuraçãoViagem")
    plt.xlabel("DuraçãoViagem")
    plt.ylabel(coluna)
    plt.show()

# Isoladamente, nao se verifica nenhuma relação linear com a duração da viagem

# transformando as variaveis qualitativas em quantitativas

# Selecionar apenas as colunas não numéricas
cols_to_encode = df.select_dtypes(exclude='number').columns

# Criar um dicionário para armazenar os mapeamentos
label_dict = {}

# Iterar sobre as colunas selecionadas e aplicar o LabelEncoder
for col in cols_to_encode:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])
    label_dict[col] = {label: category for label, category in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

print(label_dict)

correlation_matrix = df.corr()

# Configurar o tamanho da figura
plt.figure(figsize=(10, 8))

# Plotar o mapa de calor da matriz de correlação
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)

# Configurar o título do gráfico
plt.title('Matriz de Correlação')

# Exibir o gráfico
plt.show()

#Distribuição do Total de Giros
sns.displot(data=df, x='TotalGiros')

plt.title(f"Distribuição dos Dados da TotalGiros:")
plt.xlabel('TotalGiros')
plt.ylabel("Contagem")

sns.displot(data=df, x='NoVeículo')

plt.title(f"Distribuição dos Dados da NoVeículo:")
plt.xlabel('NoVeículo')
plt.ylabel("Contagem")

#Distribuição do Total de Giros
sns.displot(data=df, x='KmPerc')

plt.title(f"Distribuição dos Dados do KmPerc:")
plt.xlabel('KmPerc')
plt.ylabel("Contagem")

# Criando faixas de separação por regras de sturges.

num_bins = int(np.ceil(np.log2(len(df['DuraçãoViagem'])) + 1))

# Calcular o tamanho de cada faixa
bin_size = (df['DuraçãoViagem'].max() - df['DuraçãoViagem'].min()) / num_bins

# Criar as faixas
bins = [df['DuraçãoViagem'].min() + i * bin_size for i in range(num_bins + 1)]

# Criar a nova coluna 'STURGES' com as faixas correspondentes
df['STURGES'] = pd.cut(df['DuraçãoViagem'], bins=bins)

df.head()

"""#**Construção do modelo**"""

# Criando a regressão linear
X = df[['DiaSemana', 'HoraIni', 'Linha', 'Sentido', 'NoVeículo', 'TotalGiros', 'KmPerc']]
y = df['DuraçãoViagem']

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

# O modelo explicou 62.3 % da variabilidade dos dados
print(results.summary())

# Verificando se os residúos estão com variância constante.
# Os resíduos precisam estar aleatoriamente distribuidos

# Calcular os resíduos
residuals = results.resid

# Plotar os resíduos
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(results.fittedvalues, residuals)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Valores ajustados')
ax.set_ylabel('Resíduos')
ax.set_title('Gráfico de Resíduos')

# Exibir o gráfico
plt.show()

# Calcular os resíduos
residuals = results.resid

# Plotar o histograma dos resíduos
plt.hist(residuals, bins=100)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Histograma dos Resíduos')
plt.show()

# Construção do modelo sem outlier

# Calcular os valores de quartis e IQR
Q1 = df['DuraçãoViagem'].quantile(0.25)
Q3 = df['DuraçãoViagem'].quantile(0.75)
IQR = Q3 - Q1

# Definir os limites para identificação de outliers
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Filtrar os outliers no novo DataFrame
d_no_outliers = df[(df['DuraçãoViagem'] >= lower_limit) & (df['DuraçãoViagem'] <= upper_limit)]

# Exibir o novo DataFrame sem os outliers
len(d_no_outliers)

# Criando a regressão linear
X = d_no_outliers[['DiaSemana', 'HoraIni', 'Linha', 'Sentido', 'NoVeículo', 'TotalGiros', 'KmPerc']]
y = d_no_outliers['DuraçãoViagem']

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

# O modelo explicou 0.649 % da variabilidade dos dados
print(results.summary())

# Calcular os resíduos
residuals = results.resid

# Plotar o histograma dos resíduos
plt.hist(residuals, bins=100)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Histograma dos Resíduos')
plt.show()

# Aplicando a transformação de box-cox em Y para tentar deixar os resíduos normais.

# Aplicar a transformação de Box-Cox em y
transformed_y, lambda_value = stats.boxcox(y)
transformed_y, lambda_value

# Criando um dataframe 2 com y transformado
df.head()

df2 = df.copy()  # Cria uma cópia do DataFrame original.
df2['DuraçãoViagem'] = df2['DuraçãoViagem'] ** 0.39868456
df2.head()

# Criando a regressão linear
X = df2[['DiaSemana', 'HoraIni', 'Linha', 'Sentido', 'NoVeículo', 'TotalGiros', 'KmPerc']]
y = df2['DuraçãoViagem']

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

print(results.summary())

# Verificando se os residúos estão com variância constante.
# Os resíduos precisam estar aleatoriamente distribuidos

# Calcular os resíduos
residuals = results.resid

# Plotar os resíduos
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(results.fittedvalues, residuals)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Valores ajustados')
ax.set_ylabel('Resíduos')
ax.set_title('Gráfico de Resíduos')

# Exibir o gráfico
plt.show()

# Os resíduos mostram que faltam variáveis no modelo para explicar a variabilidade dos dados
# A reta em baixo

# Calcular os resíduos
residuals = results.resid

# Plotar o histograma dos resíduos
plt.hist(residuals, bins=100)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Histograma dos Resíduos')
plt.show()

len(df2)

# Criando um novo modelo sem outlier

# Calcular os valores de quartis e IQR
Q1 = df2['DuraçãoViagem'].quantile(0.25)
Q3 = df2['DuraçãoViagem'].quantile(0.75)
IQR = Q3 - Q1

# Definir os limites para identificação de outliers
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Filtrar os outliers no novo DataFrame
df3 = df2[(df2['DuraçãoViagem'] >= lower_limit) & (df2['DuraçãoViagem'] <= upper_limit)]

# Exibir o novo DataFrame sem os outliers
len(df3)

# 218917 - 215570 quantidade de outliers.

# Construção do modelo sem outlier

# Calcular os valores de quartis e IQR
Q1 = df3['TotalGiros'].quantile(0.25)
Q3 = df3['TotalGiros'].quantile(0.75)
IQR = Q3 - Q1

# Definir os limites para identificação de outliers
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Filtrar os outliers no novo DataFrame
d_no_outliers = df3[(df3['TotalGiros'] >= lower_limit) & (df3['TotalGiros'] <= upper_limit)]

# Exibir o novo DataFrame sem os outliers
len(d_no_outliers)

# Criando a regressão linear sem outlier
X = df_no_outliers[['DiaSemana', 'HoraIni', 'Linha', 'Sentido', 'NoVeículo', 'TotalGiros', 'KmPerc']]
y = df_no_outliers['DuraçãoViagem']

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

print(results.summary())
# Explicou 0.689 (Modelo escolhido)
# Y_novo = y ^ 3986 (transformação de boxcox)
# O dia da semana não se mostrou significativo.

"""#**Modelo de regressão escolhido**"""

# Criando a regressão linear sem outlier e dia da semana

X = df_no_outliers[[ 'HoraIni', 'Linha', 'Sentido', 'NoVeículo', 'TotalGiros', 'KmPerc']]
y = df_no_outliers['DuraçãoViagem']

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

print(results.summary())

# Calcular os resíduos
residuals = results.resid

# Plotar o histograma dos resíduos
plt.hist(residuals, bins=100)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Histograma dos Resíduos')
plt.show()

"""#**Construção do modelo de classificação**"""

d_no_outliers.head()

d_no_outliers2 = d_no_outliers.copy()
d_no_outliers = d_no_outliers.dropna()

# Aplicar o LabelEncoder
encoder = LabelEncoder()
d_no_outliers['sturges_encoded'] = encoder.fit_transform(d_no_outliers['STURGES'])

# Obtém os códigos numéricos únicos
codigos_numericos = d_no_outliers['sturges_encoded'].unique()

# Obtém os significados originais dos códigos
significados = encoder.inverse_transform(codigos_numericos)

# Cria um dicionário para mapear os códigos numéricos aos significados
mapeamento = dict(zip(codigos_numericos, significados))

X = d_no_outliers.drop(['DuraçãoViagem','STURGES','sturges_encoded'], axis=1)
y = d_no_outliers['sturges_encoded']

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Ou RandomForestRegressor para variável numérica
from sklearn.metrics import classification_report  # Ou outras métricas relevantes

# Separar os dados em conjunto de treinamento e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo RandomForest.
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Ajustar o modelo aos dados de treinamento.
modelo_rf.fit(X_train, y_train)

# Fazer previsões usando o modelo.
y_pred = modelo_rf.predict(X_test)

print(classification_report(y_test, y_pred))

"""#**Regressor - Duração da Viagem**"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error
from sklearn.metrics import mean_squared_error,median_absolute_error

d_no_outliers2

X = df_no_outliers[[ 'HoraIni', 'Linha', 'Sentido', 'TotalGiros', 'KmPerc','DiaSemana']]
y = df_no_outliers['DuraçãoViagem']

# Separar os dados em conjunto de treinamento e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y, test_size=0.3, random_state=42)

# Criar o modelo baseado em histogramas.
modelo_hist = HistGradientBoostingRegressor(categorical_features=['Linha','Sentido','DiaSemana'])

# Ajustar o modelo aos dados de treinamento.
modelo_hist.fit(X_train, y_train)

modelo_hist.score(X_train, y_train)

# Fazer previsões usando o modelo.
y_pred = modelo_hist.predict(X_test)

print(f'Coeficiente R2: {r2_score(y_test,y_pred)}\n')
print(f'Variância explicada: {explained_variance_score(y_test,y_pred)}\n')
print(f'Mean Absolute Error: {mean_absolute_error(y_test,y_pred)}\n')
print(f'Mean Squared Error: {mean_squared_error(y_test,y_pred)}\n')
print(f'Median Absolute Error: {median_absolute_error(y_test,y_pred)}\n')

"""#**Regressor - Total de Giros**"""

from sklearn.ensemble import RandomForestRegressor

d_no_outliers2

X = df_no_outliers[[ 'HoraIni', 'Linha', 'Sentido', 'DuraçãoViagem', 'KmPerc','DiaSemana']]
y = df_no_outliers['TotalGiros']

# Separar os dados em conjunto de treinamento e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo baseado em histogramas.
modelo_hist2 = HistGradientBoostingRegressor(categorical_features=['Linha','Sentido','DiaSemana'])
modelo_rfor = RandomForestRegressor(random_state=0)  # Ou RandomForestRegressor para variável numérica.

# Ajustar o modelo aos dados de treinamento.
modelo_hist2.fit(X_train, y_train)
modelo_rfor.fit(X_train, y_train)

modelo_hist2.score(X_train, y_train)

modelo_rfor.score(X_train, y_train)

y_pred = modelo_hist2.predict(X_test)
y_pred2 = modelo_rfor.predict(X_test)

print("HISTOGRAMA\n")
print(f'Coeficiente R2: {r2_score(y_test,y_pred)}\n')
print(f'Variância explicada: {explained_variance_score(y_test,y_pred)}\n')
print(f'Mean Absolute Error: {mean_absolute_error(y_test,y_pred)}\n')
print(f'Mean Squared Error: {mean_squared_error(y_test,y_pred)}\n')
print(f'Median Absolute Error: {median_absolute_error(y_test,y_pred)}\n')

print("RANDOM FOREST\n")
print(f'Coeficiente R2: {r2_score(y_test,y_pred2)}\n')
print(f'Variância explicada: {explained_variance_score(y_test,y_pred2)}\n')
print(f'Mean Absolute Error: {mean_absolute_error(y_test,y_pred2)}\n')
print(f'Mean Squared Error: {mean_squared_error(y_test,y_pred2)}\n')
print(f'Median Absolute Error: {median_absolute_error(y_test,y_pred2)}\n')
