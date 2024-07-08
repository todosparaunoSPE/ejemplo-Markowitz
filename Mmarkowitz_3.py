# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:26:55 2024

@author: jperezr
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuración de la aplicación
st.title('Análisis de Portafolios - Modelo de Markowitz')
st.header("Creado por: Javier Horacio Pérez Ricárdez")

st.write("")
st.write("")

# Sección de ayuda en la barra lateral
st.sidebar.title("Ayuda")
st.sidebar.write("""
Esta aplicación permite analizar portafolios de inversión utilizando el Modelo de Markowitz. 
A continuación, se detallan los pasos y funcionalidades:

1. **Entrada de Tickers**: Ingrese las abreviaturas de los indicadores bursátiles (tickers) que desea analizar. 
   Ejemplo: KO,TLT,LQD,SPY,AAPL.

2. **Selección de Fechas**: Especifique el rango de fechas para el análisis.

3. **Descarga de Datos**: La aplicación descarga datos de precios ajustados de Yahoo Finance para los tickers y el rango de fechas seleccionados.

4. **Cálculo de Rendimientos Diarios**: Se calculan los rendimientos diarios de los precios ajustados.

5. **Rendimiento Esperado**: Se calcula el rendimiento esperado (media de los rendimientos diarios).

6. **Matriz de Covarianza**: Se calcula la matriz de covarianza de los rendimientos.

7. **Simulación de Portafolios**: Se simulan 10,000 portafolios aleatorios para encontrar la mejor combinación de retorno y riesgo.

8. **Resultados de las Simulaciones**: Se muestran los resultados de las simulaciones, incluyendo el portafolio con la mayor relación de Sharpe.

9. **Gráfico de Dispersión**: Se visualizan los portafolios simulados en un gráfico de riesgo vs rendimiento, destacando el portafolio óptimo.
""")

# Sección de derechos de autor
st.sidebar.write("")
st.sidebar.write("© 2024 Todos los derechos reservados. Creado por jahoperi")

# Entrada de tickers
st.write("Los tickers son las abreviaturas de los indicadores, se pueden consultar en Yahoo Finance")

st.markdown('[Visita Yahoo Finance](https://finance.yahoo.com/)')

tickers_input = st.text_input(
    'Ingrese los tickers (indicadores) separados por comas (ej. KO,TLT,LQD,SPY,AAPL)', 
    'KO,TLT,LQD,SPY,AAPL'
)
tickers = [ticker.strip() for ticker in tickers_input.split(',')]

# Selección de fechas
start_date = st.date_input('Fecha de inicio', datetime(2020, 1, 1))
end_date = st.date_input('Fecha de fin', datetime(2023, 1, 1))

if start_date >= end_date:
    st.error('La fecha de inicio debe ser anterior a la fecha de fin.')
else:
    # Descargar datos de precios ajustados de Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']

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
    st.write("el promedio de las columnas de la tabla anterior")
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

    # Mostrar DataFrame con los resultados de las simulaciones
    st.subheader("Resultados de las 10,000 simulaciones de portafolios")
    st.subheader("de donde se obtiene la cartera óptima")
    st.dataframe(results_df)

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
