import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

dados = pd.read_csv("Dados.csv")

#trabalhando com a segunda coluna de dados do arquivo
X = np.array(dados.index.values).reshape(-1, 1)
y = np.array(dados.iloc[:, 1]).reshape(-1, 1)

#treinando 
reg = LinearRegression().fit(X, y)

#prevendo os próximos 5 dias
tratandoDados = np.array(range(len(dados), len(dados) + 5)).reshape(-1, 1)
previsao = reg.predict(tratandoDados)

print(previsao)
