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
# Leer una dataframe
df = pd.read_csv('./data/BD_Inicial_Stat.csv', encoding='latin-1')

def BD_def():
    st.write("---")  # Línea divisoria horizontal
    # Titulo
    st.header("Base de Datos Definitiva:")
    st.subheader("Descripción Estadística")
    st.dataframe(df)
 
    st.write("---")  # Línea divisoria horizontal
    st.header("Series de Tiempo")

    # Insertar una imagen a nuestra página
    st.subheader("Oceanic Niño Index")
    imagen = Image.open('media/var/20_líneas_ONI.png')
    st.image(imagen, caption="Serie de tiempo del Oceanic Niño Index (Datos: NOAA, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Volúmenes de Embalses")
    imagen = Image.open('media/var/02_Líneas_Volumenes_Embalses.png')
    st.image(imagen, caption="Serie de tiempo de Volúmenes de Embalses (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Generación con Fósiles")
    imagen = Image.open('media/var/04_Líneas_Generación_Fósiles.png')
    st.image(imagen, caption="Serie de tiempo de Generación con Fósiles (Sinergox, 2024)",
             use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Generación Hidráulica")
    imagen = Image.open('media/var/06_Líneas_Generación_Hidráulica.png')
    st.image(imagen, caption="Serie de tiempo de Generación Hidráulica (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Generación con Biomasa")
    imagen = Image.open('media/var/08_Líneas_Generación_Biomasa.png')
    st.image(imagen, caption="Serie de tiempo de Generación con Biomasa (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Generación Eólica y Solar")
    imagen = Image.open('media/var/10_Líneas_Generación_Eólica_Solar.png')
    st.image(imagen, caption="Serie de tiempo de Generación Eólica y Solar (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Precio del Gas Natural")
    imagen = Image.open('media/var/22_Líneas_Precio_Gas_Natural.png')
    st.image(imagen, caption="Serie de tiempo del Precio del Gas Natural (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Confiabilidad OEF")
    imagen = Image.open('media/var/12_Líneas_Obligación_Energía_Firme_OEF.png')
    st.image(imagen, caption="Serie de tiempo de Confiabilidad OEF (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Tasa Representativa del Mercado del Dolar Americano")
    imagen = Image.open('media/var/16_Líneas_TRM.png')
    st.image(imagen, caption="Serie de tiempo de la TRM (SFC-Datos Abiertos, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Confiabilidad RRID")
    imagen = Image.open('media/var/14_Líneas_Confiabilidad_RRID.png')
    st.image(imagen, caption="Serie de tiempo de la Confiabilidad RRID (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Interacción con Ecuador")
    imagen = Image.open('media/var/18_Líneas_Interacción_Ecuador.png')
    st.image(imagen, caption="Serie de tiempo de Interacción con Ecuador (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Precio de Energía en Bolsa")
    imagen = Image.open('media/var/25_Líneas_Precio_Energía_Bolsa.png')
    st.image(imagen, caption="Serie de tiempo del Precio de Energía en Bolsa (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Precio del Gas Natural vs ONI")
    imagen = Image.open('media/var/23_Comparación_Precio_Gas_Natural_vs_ONI.png')
    st.image(imagen, caption="Serie de tiempo de Precio del Gas Natural vs ONI (Sinergox, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

    # Insertar una imagen a nuestra página
    st.subheader("Precio de la Energía vs ONI")
    imagen = Image.open('media/var/26_Comparación_Precio_Energía_Bolsa_vs_ONI.png')
    st.image(imagen, caption="Serie de tiempo del Precio de la Energía vs ONI (Datos: NOAA, 2024)",
            use_container_width=600)
    st.write("---")  # Línea divisoria horizontal

BD_def()


