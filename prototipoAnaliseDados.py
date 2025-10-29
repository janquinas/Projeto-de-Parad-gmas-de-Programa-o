import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# --- 1. Dados E CÁLCULO DE INDICADORES ---
dados = {
    "dia": [1, 2, 3, 4, 5, 6, 7],
    "pecas_totais": [160, 90, 260, 180, 385, 140, 150],
    "pecas_boas": [137, 83, 197, 140, 268, 130, 135],
}
df = pd.DataFrame(dados)

# Calculando peças defeituosas
df["pecas_defeituosas"] = df["pecas_totais"] - df["pecas_boas"]

# Calculando indicadores
df["produtividade"] = df["pecas_boas"] / df["pecas_totais"]
df["taxa_defeitos"] = df["pecas_defeituosas"] / df["pecas_totais"]

produtividade_media = df["produtividade"].mean()
taxa_defeitos_media = df["taxa_defeitos"].mean()

print("--- Indicadores Descritivos ---")
print(f"Produtividade média: {produtividade_media:.2%}")
print(f"Taxa de defeitos média: {taxa_defeitos_media:.2%}")


# --- 2. ANÁLISE PREDITIVA COM "IA" ---
print("\n--- Iniciando Análise Preditiva ('IA') ---")

# Preparando dados para o modelo (Usando Numpy)
# O modelo (sklearn) espera os dados de entrada (X) como um array 2D
# e os dados de saída (y) como um array 1D.
# Usamos .values para obter os arrays numpy do DataFrame pandas.
X = df[["dia"]].values  # Features (Dia)
y = df["taxa_defeitos"].values  # Target (o que vai ser previsto)

# Treinando o modelo de Regressão Linear
modelo_ia = LinearRegression()
modelo_ia.fit(X, y)

# Fazendo uma previsão para o próximo dia (dia 8)
dia_previsao = 8
# Usamos np.array para criar a entrada da previsão no formato 2D correto
previsao_dia_8 = modelo_ia.predict(np.array([[dia_previsao]]))

print(f"Previsão da Taxa de Defeitos (IA) para o Dia {dia_previsao}: {previsao_dia_8[0]:.2%}")

# Gerando a linha de tendência (previsões do modelo para os dias existentes)
y_tendencia = modelo_ia.predict(X)


# --- 3. GERANDO GRÁFICO ---

plt.figure(figsize=(10, 6))
plt.plot(df["dia"], df["produtividade"], marker="o", label="Produtividade (Real)")
plt.plot(df["dia"], df["taxa_defeitos"], marker="x", label="Taxa de Defeitos (Real)", color="red")

# Linha de tendência ("IA") ao gráfico
plt.plot(df["dia"], y_tendencia, linestyle="--", color="green", label="Tendência Defeitos (IA)")

plt.title("Indicadores de Produção com Tendência de IA")
plt.xlabel("Dia")
plt.ylabel("Taxa")
plt.legend()
plt.grid(True)
# Salvando com novo nome
plt.savefig("grafico_indicadores.png")
plt.close()


# --- 4. GERANDO RELATÓRIO PDF (Modificado para incluir a "IA") ---

# Salvando com novo nome
c = canvas.Canvas("relatorio_producao.pdf", pagesize=A4)
c.setFont("Helvetica-Bold", 16)
c.drawString(100, 800, "Relatório de Produção")

c.setFont("Helvetica", 12)
c.drawString(100, 770, f"Produtividade média: {produtividade_media:.2%}")
c.drawString(100, 750, f"Taxa de defeitos média: {taxa_defeitos_media:.2%}")

# Adicionando a previsão da "IA" ao PDF
c.setFont("Helvetica-Bold", 12)
c.drawString(100, 720, "Previsão do Modelo de IA:")
c.setFont("Helvetica", 12)
c.drawString(100, 700, f"Taxa de Defeitos prevista p/ Dia {dia_previsao}: {previsao_dia_8[0]:.2%}")

# Inserindo gráfico no PDF
c.drawImage("grafico_indicadores.png", 100, 400, width=450, height=270)

c.save()

print(f"\nRelatório PDF ('relatorio_producao.pdf') e Gráfico ('grafico_indicadores.png') gerados com sucesso.")