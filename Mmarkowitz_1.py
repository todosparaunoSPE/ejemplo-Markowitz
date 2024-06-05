# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:13:30 2024

@author: jperezr
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuración de la aplicación
st.title('Análisis de Portafolios')

st.subheader("")

# Entrada de tickers
st.write("Los tickers son las abrebiaturas de los indicadores, consultalos en yahoo finance")

st.markdown('[Visita yahoo finance](https://finance.yahoo.com/)')

tickers_input = st.text_input('Ingrese los tickers (inidcadores) separados por comas (ej. KO,NEM,NRG,NVDA)', 'KO,NEM,NRG,NVDA')
tickers = [ticker.strip() for ticker in tickers_input.split(',')]

# Selección de fechas
start_date = st.date_input('Fecha de inicio', datetime(2020, 1, 1))
end_date = st.date_input('Fecha de fin', datetime(2023, 1, 1))

if start_date >= end_date:
    st.error('La fecha de inicio debe ser anterior a la fecha de fin.')
else:
    # Descargar datos de precios ajustados de Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Crear DataFrame con los datos descargados
    df = pd.DataFrame(data)

    # Mostrar el DataFrame en Streamlit
    st.write(f"Datos de precios ajustados de Yahoo Finance del {start_date} a {end_date}:")
    st.dataframe(df)

    st.subheader("")

    # Calcular rendimientos diarios
    returns = data.pct_change().dropna()

    # Mostrar el DataFrame en Streamlit
    st.write("Datos de los rendimientos diarios:")
    st.write(" (valor del día siguiente - valor del día anterior) / valor del día anterior")
    st.dataframe(returns)

    # Calcular el rendimiento esperado (media de rendimientos diarios)
    mean_returns = returns.mean()
    mean_df = pd.DataFrame(mean_returns, index=tickers)

    st.subheader("")
    st.write("Matriz de rendimiento esperado (media de rendimientos diarios):")
    st.write("el promedio de las columnas de la tabla anteior")
    st.dataframe(mean_df)

    st.subheader("")

    # Calcular la matriz de covarianza de los rendimientos
    cov_matrix = returns.cov()
    cov_df = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)

    st.write("Matriz de Covarianza de Rendimientos:")
    st.dataframe(cov_df)

    # Simulación de portafolios
    num_portfolios = 10000
    results = np.zeros((3 + len(tickers), num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = results[0, i] / results[1, i]
        for j in range(len(tickers)):
            results[3 + j, i] = weights[j]

    columns = ['Return', 'StdDev', 'Sharpe'] + [f'PESO_{ticker}' for ticker in tickers]
    results_df = pd.DataFrame(results.T, columns=columns)

    max_sharpe_idx = results_df['Sharpe'].idxmax()
    max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]

    st.subheader("")
    st.write("Portafolio con la mayor relación de Sharpe:")
    st.write("de las 10,000 simulaciones (puntos de riesgo vs rendimiento esperado del gráfico)")
    st.write("se elige el de mayor ratio Sharpe")
    st.write(max_sharpe_portfolio)

    # Gráfico de dispersión
    st.subheader("")
    plt.scatter(results_df['StdDev'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis')
    plt.xlabel('Riesgo (Desviación Estándar)')
    plt.ylabel('Rendimiento Esperado')
    plt.colorbar(label='Relación de Sharpe')
    plt.scatter(max_sharpe_portfolio['StdDev'], max_sharpe_portfolio['Return'], marker='*', color='r', s=200)
    st.pyplot(plt)