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
st.write("---")
st.write("#### BootCamp:")
st.write("""
El proyecto PEBC se desarrolló usando la metodología de BootCamp. Esta es una forma de aprendizaje intensivo y práctico enfocada resolver problemas complejos de desarrollo de software. Se realizó en forma colaborativa en un equipo de cuatro personas con diferentes habilidades y competencias, el cual se describe a continuación:\n
- Programa:         Talento Tech 2.0
- Región 2:	        Antioquia, Caldas, Chocó, Quindío, Risaralda
- Curso:            VIRTA03-1-Análisis de datos Innovador - Avanzado-2024-3-L1-G3\n           
- Promotor:         Ministerio de Tecnologías de la Información y las Comunicaciones\n 
- Ejecutores:       Universidad de Antioquia, Universidad de Caldas.
- Soporte:          Ubicua Technology y UI Training
- Modalidad:        Virutal\n
- Duración:         8 semanas\n
- Fecha de inicio:  octubre 21, 2024\n
- Fecha de cierre:  diciembre 13, 2024\n
- Intensidad:       20 horas/semana (total 160 horas)
""")
st.write("---")
