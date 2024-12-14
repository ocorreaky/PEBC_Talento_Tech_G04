import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import GridSearchCV # Búsqueda de cuadrículas
from pickle import dump
from pickle import load
import pickle as pkl
import os
import streamlit as st
import utilidades as util

# Activar entorno: .\.venv\Scripts\activate
# (env) PS D:\Ejemplo_Streamlit> streamlit run ./Home.py
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

# Configuración Apertura
util.Config_pag()

# Llamar el menú
util.generarMenu()

# Redimensionar los logotipos
logo1 = util.resize_image("media/logos/logo1.png", 2.00)
logo2 = util.resize_image("media/logos/logo2.png", 2.00)
logo3 = util.resize_image("media/logos/logo3.png", 2.00)

util.Logos_y_Titulo(logo1, logo2, logo3)

############################################################################################
# Crear el texto
st.write("---")  # Línea divisoria horizontal
st.write('#### ¿Qué es PEBC?:')
st.markdown("""
***PEBC*** es un conjunto de modelos avanzados de **análisis de datos**, que incluye **métodos multivariados* de **regresión lineal y no lineal** con **machine learning**, **optimización**, así como herramientas de **visualización** de alto nivel, para intentar resolver el problema complejo de **estimar el precio de venta de la energía en la Bolsa de Colombia**. 

Se basa en datos de **almacenamiento hidráulico**, **matriz de generación denergética**, **confiabilidad del sistema**, **intercambio comercial con Ecuador**, **precio del dolar en Colombia** e **índices de cambio climático**. 

Esta solución innovadora ofrece **resultados de referencia** para **monitorear los precios** y las **posibles inelasticidades** del **mercado de la energía** en bolsa, favoreciendo la **toma de decisiones** durante la negociación de un **precio justo** para **generadores y consumidores**.

Responde a **lineamientos** del **Ministerio de Tecnologías de la Información y las Comunicaciones** (MINTIC) de Colombia relacionados con la ***Democratización de la Generación y el Consumo Energético***, dentro del programa de capacitación **Talento Tech 2.0**.
"""
         )

# Videos
st.write("---")  # Línea divisoria horizontal
st.write("¿Cómo funciona la compra de energía en Colombia para abastecer la demanda nacional?")
with open("./media/vid/Compra Energía Bolsa.mp4", "rb") as video_file:
        st.video(video_file.read(), start_time=0)
    
