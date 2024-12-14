import utilidades as util
from utilidades import *
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

def Estructura_Prueba_Modelos():
    st.write("---")  # Línea divisoria horizontal
    st.header("Prueba de Variables y Modelos:")
    st.subheader("Parámetros Definidos por el Usuario")
    
    # Llama a la función Seleccion_Modelo() => str
    modelo_deseado = util.Seleccion_Modelo()
    st.markdown(f"<p style='color: darkslategrey;'>El modelo seleccionado es: {modelo_deseado}</p>\n", 
            unsafe_allow_html=True)

    # Cambiar la paleta de color según selección específica
    color, colores = util.SeleccionPaleta(modelo_deseado)
    st.write("\n")  # Línea divisoria horizontal

    # Cambiar el color del texto principal
    stile_color = util.set_custom_style(color)
    
    # Llama a la función Seleccion_Variables() => list
    descartar = util.Seleccion_Descartar()

    if descartar == "ninguna":
        st.markdown("<p style='color: darkslategrey;'>No se descartará ninguna variable.</p>\n", 
            unsafe_allow_html=True)
    else:
        if descartar == []:
            pass
        else:
            st.markdown(f"<p style='color: darkslategrey;'>Eligió descartar: {descartar}</p>\n", 
            unsafe_allow_html=True)
    st.write("\n")  # Línea divisoria horizontal

    # Armar el dataframe eliminando descartadas
    considerar = util.Seleccion_Considerar(descartar)
    y_considerar = df['y']
    df1 = Armado_DataFrame(considerar, y_considerar, df)
    
    # Rando de datos para ajustar los modelos 
    v_min = 500 # número mínimo de datos para desarrollar el modelo
    v_max = len(df1)

    # Llama a la función de slider continuo para definir el número de datos
    num_data = util.Seleccion_Datos(v_min, v_max)
    st.markdown(f"<p style='color: darkslategrey;'>El número de datos elegido es: {num_data}</p>\n", 
            unsafe_allow_html=True)
    st.write("\n")  # Línea divisoria horizontal

    # Hacer muestreo aleatorio del df1 según número de datos seleccionado (usa semilla)
    df_def = df1.sample(n=num_data, random_state=42)

    # Selecciona la unidad de tiempo para los pronósticos
    frec_temp = util.Seleccion_Unidad_Tiempo()
    st.markdown(f"<p style='color: darkslategrey;'>la unidad de predicción elegida es: {frec_temp}</p>\n", 
            unsafe_allow_html=True)
    st.write("---")  # Línea divisoria horizontal

    # Insertar dataframe df_def con BD definitiva para ajuste de modelos
    st.subheader("Variables de entrada y Salida para análisis:")
    st.dataframe(df_def, use_container_width=True)
    st.write('Número de registros:', len(df_def), 'Número de variables de entrada:', len(df_def.columns)-1)
    st.write("---")  # Línea divisoria horizontal

    # Generar matriz de correlación y dibujar el mapa de calor
    st.write("### Correlaciones entre Variables de Entrada Seleccionadas:")
    
    # Separar las variables de entrada y salida
    X = df_def.drop(columns=['y'])
    y = df_def['y']
 
    # Crear el mapa de calor
    util.Creacion_Mapa_Calor(X, colores)
    st.write("---")  # Línea divisoria horizontal

    # Separar los conjuntos de datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = util.Set_Data_train_test(X, y)
    
    # Ajustar el modelo
    model = util.Ajuste_Modelo(X_train, X_test, y_train, y_test, modelo_deseado)

    # Predicciones
    y_train_pred, y_test_pred = util.Predic_Model(X_train, X_test, model)

    # Evaluación del modelo
    train_rmse, test_rmse, train_r2, test_r2, residuals = util.Eval_Model(y_train, y_test, y_train_pred, y_test_pred)

    # Crear la gráfica de dispersión de datos vs modelo ajustado
    st.write("### Dispersión de Datos Reales:")
    util.Fig_Ajuste_Modelo(y_test, y_test_pred, y_train_pred, X, color, modelo_deseado)
    st.write("---")  # Línea divisoria horizontal

    # Crear el histograma de residuos 
    util.Fig_Distrib_Residuos(residuals, X, color, modelo_deseado)
    st.write("---")  # Línea divisoria horizontal

    # Crea la tabla de Resultados
    st.write("### Resultados del modelo:")
    util.Tabla_Resultados_Ajuste(train_rmse, test_rmse, train_r2, test_r2, residuals)
    st.write("---")  # Línea divisoria horizontal

    # Crea la gráfica de linea de tiempo
    st.write("### Linea de Tiempo Valores Reales vs. Pronóstico:")
    util.Fig_Linea_Tiempo_Model(y_test, y_test_pred, modelo_deseado)
    st.write("---")  # Línea divisoria horizontal
    
    # Guardado de modelos
    num_var = len(considerar)
    num_data_fig = len(y_train)
    path = 'models/'
    filename = f'mod_{modelo_deseado}_{num_data_fig}_{num_var}.sav'
    #filename = util.Guardado_Modelos(modelo_deseado, num_data_fig, num_var, model)
    rta = 0
    rta = st.button('Guardar Modelo', on_click=lambda: pkl.dump(model, open(path + filename, 'wb')))
    if rta:
        st.info(f"NOTA:\nEl modelo:    {filename}     ................ fue guardado con éxito!")
 

Estructura_Prueba_Modelos()

