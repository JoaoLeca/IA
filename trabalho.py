import pandas as pd #Manipulação de dados (tabelas)
from sklearn.model_selection import train_test_split # Divide dados em treino/teste
from sklearn.linear_model import LinearRegression #Cria modelo de regressão linear
from sklearn.metrics import mean_squared_error, r2_score # Mede o erro médio ao quadrado #Mede a qualidade da predição
from sklearn.preprocessing import LabelEncoder #Converte texto para número

# 1. Carregar a planilha CSV
df = pd.read_csv('planilha_IMC_PesoIdeal_500_pessoas(Sheet1).csv')

# 2. Calcular o IMC (target para regressão)
df['IMC'] = df['Peso_kg'] / (df['Altura_m'] ** 2)

# 3. Codificar o sexo (transformar string em número)
le = LabelEncoder()
df['Sexo_codificado'] = le.fit_transform(df['Sexo'])  # Masculino=1, Feminino=0

# 4. Separar as variáveis de entrada (X) e saída (y)
X = df[['Altura_m', 'Peso_kg', 'Sexo_codificado']]
y = df['IMC']

# 5. Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Criar e treinar o modelo de regressão
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 7. Fazer previsões
y_pred = modelo.predict(X_test)

# 8. Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro médio quadrático (MSE): {mse:.2f}")
print(f"Coeficiente de determinação (R²): {r2:.2f}")

# 9. Exibir os coeficientes do modelo
print("\nCoeficientes do modelo:")
for col, coef in zip(X.columns, modelo.coef_):
    print(f"{col}: {coef:.2f}")

# 10. Adicionar os resultados ao DataFrame original (apenas para os dados de teste)
df_test = X_test.copy()
df_test['IMC_real'] = y_test.round(0).astype(int)  # Arredonda para inteiro

# Adicionar colunas 'Nome' e 'Sexo' com base nos índices originais
df_test['Sexo'] = df.loc[df_test.index, 'Sexo']
df_test['Nome'] = df.loc[df_test.index, 'Nome']

# Adicionar a coluna de categoria de peso baseada no IMC real (arredondado)
def categorizar_imc(imc):
    if imc < 18.5:
        return 'Abaixo do normal'
    elif imc < 25:
        return 'Normal'
    elif imc < 30:
        return 'Sobrepeso'
    elif imc < 35:
        return 'Obesidade grau I'
    elif imc < 40:
        return 'Obesidade grau II'
    else:
        return 'Obesidade grau III'

df_test['Categoria_Peso'] = df_test['IMC_real'].apply(categorizar_imc)

# Reordenar e selecionar as colunas desejadas
df_resultado = df_test[['Nome', 'Sexo', 'Altura_m', 'Peso_kg', 'IMC_real', 'Categoria_Peso']]

# Exibir os resultados
print("\nResultados com nomes, sexo, IMC real arredondado e categoria de peso:")
print(df_resultado.to_string(index=False))

# Função para exportar resultados para Excel
def exportar_para_excel(dataframe, nome_arquivo='relatorio_IMC_resultados.xlsx'):
    try:
        dataframe.to_excel(nome_arquivo, index=False)
        print(f"\n✅ Arquivo Excel exportado com sucesso: {nome_arquivo}")
    except Exception as e:
        print(f"\n❌ Erro ao exportar para Excel: {e}")

# Chamar a função de exportação
exportar_para_excel(df_resultado)

# Exportar também como CSV
df_resultado.to_csv('relatorio_IMC_resultados.csv', index=False)
print("✅ Arquivo CSV exportado com sucesso: relatorio_IMC_resultados.csv")
