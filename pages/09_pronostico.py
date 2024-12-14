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
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from pickle import dump
from pickle import load
import pickle as pkl
import os
import streamlit as st
import utilidades as util

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
# Leer un dataframe
df = pd.read_csv('./data/04_datos_sin_outilers_Norm.csv')

def Pornosticar_main():
    st.write("---")  # Línea divisoria horizontal
    st.header("Pronóstico del Precio de la Energía:")

    ###### Selección del modelo para pronóstico
    # Selección del modelo de pronóstico
    model_pron = util.Seleccion_Modelo_Pronostico()

    # Mostrar el mensaje solo si se seleccionó un modelo diferente a "Ninguno"
    if model_pron and model_pron != 'Ninguno':
        st.markdown(
            f"<p style='color: darkslategrey;'>Seleccionaste: ........ {model_pron}</p>",
            unsafe_allow_html=True)
        
        # Inicializar variables
        #X_real = None
        #y_real = None
        #df_nuevos_normalizados = None

        # Selección y cargar datos para pronóstico
        df_original, df_nuevos = util.Seleccion_Nuevos_Datos()

        # Normalizar los nuevos datos
        if df_original is not None and df_nuevos is not None:
            st.markdown(
            "<p style='color: darkslategrey;'>Normalizando datos...</p>",
            unsafe_allow_html=True)
            df_nuevos_normalizados = util.Normalizar_Datos(df_original, df_nuevos)
            
            # Manejar valores NaN en los datos normalizados
            X_real = df_nuevos_normalizados.drop(columns=['y'], errors='ignore')
            if X_real.isnull().values.any():
                st.warning("Los datos contienen valores faltantes. Realizando imputación...")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')  # Reemplaza NaN con la media
                X_real = pd.DataFrame(imputer.fit_transform(X_real), columns=X_real.columns)

            y_real = df_nuevos['y']

        # Realizar predicciones
        if model_pron and model_pron != 'Ninguno' and X_real is not None:
            y_pron = util.Cargar_Modelo_y_Predecir(model_pron, X_real)
            st.write("Predicciones realizadas con éxito.")
            st.dataframe(y_pron)
        else:
            st.warning("Por favor seleccione un modelo válido y cargue datos adecuados para realizar predicciones.")

    # Crea la gráfica de linea de tiempo

    st.write("### Linea de Tiempo Valores Reales vs. Pronóstico:")
    util.Fig_Linea_Tiempo_Model(y_real, y_pron, model_pron)
    st.write("---")  # Línea divisoria horizontal



















    # Separación de variables de entrada y salida
    X = df.drop(columns=['y'])
    y = df['y']
    # Separación de Variables de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
   
    # Presentar la forma (shape) de los datos (filas, columnas)
    st.markdown('#### 1. Separación de Variables')
    st.write('Conjunto de Datos de Entrenamiento:')
    st.info(X_train.shape)
    st.write('Conjunto de Datos de Prueba:')
    st.info(X_test.shape)
    
    st.markdown('#### 2. Detalles de las Variables:')
    st.write('Variables Predictoras:')
    st.info(list(X.columns))
    st.write('Variable a Predecir')
    st.info(y.name)
       
    # Subtítulo
    st.markdown('#### 3. Características del modelo:')
    
    # Crear la instancia del bosque
    bosque = RandomForestRegressor()

    # entrenar el modelo 
    bosque.fit(X_train, y_train)

    # Hacer la predicción
    y_pred = bosque.predict(X_test)

    # Calcular métricas de evaluación
    rmse = root_mean_squared_error(y_test, y_pred)  # RMSE
    r2 = r2_score(y_test, y_pred)  # R²

    # Mostrar métricas
    st.markdown('#### 4. Métricas del Modelo:')
    st.write('**Error Cuadrático Medio (RMSE):**')
    st.info(rmse)
    st.write('**Coeficiente de Determinación (R²):**')
    st.info(r2)
    
    # Mostrar parámetros del modelo
    st.markdown('#### 5. Parámetros del modelo:')
    dict_param = bosque.get_params()

    # Convertir el diccionario a un DataFrame
    df_param = pd.DataFrame(list(dict_param.items()), columns=['Nombre', 'Valor'])

    # Mostrar el DataFrame en Streamlit
    st.dataframe(df_param, use_container_width=True)
    



Pornosticar_main()


