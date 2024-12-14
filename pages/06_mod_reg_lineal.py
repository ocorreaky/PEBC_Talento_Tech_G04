import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import utilidades as util
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

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
# Leer la base de datos norlalizada en un dataframe
df = pd.read_csv('./data/04_datos_sin_outilers_Norm.csv')

def Lineal_Model():
    st.write("---")  # Línea divisoria horizontal
    st.header("Modelo de Regresión Lineal:")
    
    # Insertar dataframe
    st.subheader("Base de Datos Normalizada:")
    st.dataframe(df)
        
    # Abrir las figuras
    imagen1 = Image.open(r'media\mod\04A_hist_var_inp_out.png')
    imagen2 = Image.open(r'media\mod\04B_Mat_corr_var_inp.png')
    imagen3 = Image.open(r'media\mod\04C_Reg_ytest_vs_ypred.png')
    imagen4 = Image.open(r'media\mod\04D_Dist_Residuos.png')
    imagen5 = Image.open(r'media\mod\04E_Analysis.png')

    # Mostrar las figuras y sus títulos
    st.write("---")  # Línea divisoria horizontal
    st.write("### Distribución de los Datos de Entrada y Salida:")
    st.image(imagen1, caption="Distribución de Fecuencias de Datos de Entrada y Salida")
    st.write("---")  # Línea divisoria horizontal
    st.write("### Análisis de correlación entre las Variables de Entrada:")
    st.image(imagen2, caption="Matriz de Correalción de las Variables de Entrada")

    st.write("---")  # Línea divisoria horizontal
    MRL = """
            # Entrenamiento del modelo
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predicciones del modelo
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Cálculo de la Raiz cuadrada del error cuadrado medio
            train_rmse = root_mean_squared_error(y_train, y_train_pred)
            test_rmse = root_mean_squared_error(y_test, y_test_pred)

            # Cálculo del Coeficiente de determinación 
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Cálculo de los residuos
            residuals = y_test - y_test_pred
    """ 
    st.write("### Código Python del Modelo:")
    st.code(MRL, language="python")
    st.write("---")  # Línea divisoria horizontal
    st.write("### Predicción de la Variable de Salida:")
    st.image(imagen3, caption="Predicciones vs. Valores Reales")
    st.write("---")  # Línea divisoria horizontal
    st.write("### Análisis de Residuos de las Predicciones:")
    st.image(imagen4, caption="Distribución de los Residuos de las Predicciones")
    st.write("---")  # Línea divisoria horizontal
    st.write("### Análisis de Resultados:")
    st.write(""" ** Evaluación de consistencia entre conjuntos de entrenamiento y prueba: **\n
    ⚠️ Diferencia significativa entre RMSE de entrenamiento y prueba.
    * RMSE de entrenamiento:\t109.8411.
    * RMSE de prueba:\t\t91.8389.

    ✅ Según R², la varianza explicada es consistente en entrenamiento y prueba (~79.5%).
    
    ** Desempeño global del modelo: **\n
    * Varianza explicada en los datos de prueba (R²):\t79.52%
    * Error medio cuadrático en datos de prueba (RMSE):\t91.8389

    ** Evaluación de residuos: **\n
    ⚠️ Los residuos no están centrados en 0. Promedio:\t-5.6166.

    ** Recomendaciones **\n
    * 🔍 El R² es moderado. 
    
    ** Considerar: **\n
    * Probar con modelos más complejos (Regresión Polinómica, Random Forest, etc.).
    * Revisar si hay variables relevantes que faltan en el modelo. """)
    
    st.write("---")  # Línea divisoria horizontal
    st.write("### Resumen de Resultados:")
    st.image(imagen5, caption="Resumen y Análisis de Resultados")

Lineal_Model()