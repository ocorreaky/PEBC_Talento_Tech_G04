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
from utilidades import *

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
st.write('### Descripción del Reto:')
st.markdown("""
El reto fue desarrollar, en un equipo de trabajo de 4 personas y bajo la metodología BootCamp, una aplicación web para estimar diariamente el precio ponderado de la energía en la Bolsa de Colombia. Este proceso se llevó a cabo en 8 semanas y abarcó los siguientes pasos:

#### 1. Estructuración de la base de datos:

* Consulta y descarga de datos.
* Exploración y análisis estadístico.
* Limpieza y preprocesamiento de datos.
* Visualización de datos.

#### 2. Desarrollo de modelos lineales y no lineales con machine learning:

* Exploración de modelos lineales y no lineales.
* Entrenamiento, prueba y optimización.
* Selección del modelo más adecuado.

#### 3. Implementación y despliegue:

* Creación de herramientas de visualización interactiva.
* Despliegue del proyecto en la web.""")

# Insertar imagen
st.write("---")  # Línea divisoria horizontal
imagen = Image.open('media/infograf/infografia_DALL_E.webp')

# Incrustar la imagen
st.image(imagen, caption="Proyecto PEBC",
         use_container_width=600)
st.write("---")  # Línea divisoria horizontal

st.write("""
Además, se fomentó el desarrollo de habilidades técnicas y habilidades de poder como comunicación asertiva, empatía, gestión del tiempo y liderazgo. La entrega final incluyó resultados, aprendizajes, y el despliegue del aplicativo, con posibles aplicaciones prácticas como emprendimientos tecnológicos o conexiones laborales.

La importancia del proyecto radica en formar capital humano para enfrentar los retos de la era digital y la inteligencia artificial, especialmente en el sector eléctrico. Los beneficios incluyen:

* Identificación anticipada de crisis energéticas.
* Regulación informada del precio de la energía.
* Apoyo en la planificación de matrices energéticas sostenibles.
* Planes de contingencia para el sector eléctrico colombiano.
""")
st.write("---")  # Línea divisoria horizontal

# Insertar una imagen a nuestra página
st.header("Modelo de Negocio")
st.write("\n")
imagen = Image.open('media/proy/WhatsApp Image 2024-12-05 at 8.57.13 PM.jpeg')
st.image(imagen, caption="Plan de Negocios",
            use_container_width=600)
st.write("---")  # Línea divisoria horizontal
