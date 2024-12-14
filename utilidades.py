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


##################################################################################################
# Utilidades Generales de la Página
##################################################################################################
def Config_pag():
    st.set_page_config(
        page_title="Precio Energía en Bolsa",
        page_icon=":dart:",
        layout="centered",
        initial_sidebar_state="auto")

def generarMenu():
    with st.sidebar:
        st.header('Navegación')
        st.page_link('Home.py', label='Inicio', icon='🏡')
        st.page_link('pages/02_reto.py', label='Reto PEBC', icon='💱')
        st.page_link('pages/03_data_original.py', label='BBDD Original', icon='📅')
        st.page_link('pages/05_data_def.py', label='BBDD Definitiva', icon='💯')
        st.page_link('pages/06_mod_reg_lineal.py', label='Modelo de Regresión Lineal', icon='💹')
        st.page_link('pages/07_analisis_var_usario.py', label='Prueba de Variables y Modelos', icon='🕹️')
        st.page_link('pages/09_pronostico.py', label='Pronóstico', icon='🎯')
        st.write("---")  # Línea divisoria horizontal
        st.page_link('pages/10_bootcamp.py', label='BootCamp', icon='⛺')
        st.page_link('pages/11_creditos.py', label='Créditos', icon='🔗')
        st.page_link('pages/12_nosotros.py', label='Nosotros', icon='📸')

# Función para redimensionar imágenes manteniendo la proporción
def resize_image(image_path, max_height_cm):
    # Asegúrate de que la ruta sea relativa al directorio raíz del proyecto
    image_path = os.path.join(os.path.dirname(__file__), image_path)
    dpi = 96  # Suposición de DPI
    max_height_px = int(max_height_cm * dpi / 2.54)
    img = Image.open(image_path)
    aspect_ratio = img.width / img.height
    new_height = max_height_px
    new_width = int(aspect_ratio * new_height)
    return img.resize((new_width, new_height))

def Logos_y_Titulo(logo1, logo2, logo3):
    # Mostrar los logotipos en una barra horizontal
    col1, _, col3, _, _, col6 = st.columns([1, 1, 1, 1, 1, 1])  # seis columnas para alinear
    with col1:
        st.image(logo1)
    with col3:
        st.image(logo2)
    with col6:
        st.image(logo3)
    # Crear el título
    st.subheader("PEBC - Precio de la Energía en la Bolsa de Colombia")

##################################################################################################
# Utilidades Página Prueba de Variables y Modelos
##################################################################################################
def Seleccion_Modelo():   # => str
    # Seleccionar el modelo deseado
    modelo_deseado = st.selectbox(
        'Seleccione el modelo deseado',
        ['Lineal', 'RandomForest', 'RejillaSCVR', 'Staking', 'XGBBOOST', 'LIGHTGBM'])
    return modelo_deseado

def SeleccionPaleta(modelo_deseado):
    dict_color = {'Lineal': 'Greens', 
                  'RandomForest': 'Blues', 
                  'RejillaSCVR': "Purples", 
                  'Staking':'Oranges',
                  'XGBBOOST': 'Reds',
                  'LIGHTGBM': 'Greys'
                  }
    colores = dict_color[modelo_deseado]
    color = colores[:-1]
    return color, colores

def set_custom_style(color_text):
    """
    Inyecta un estilo CSS personalizado en la página de Streamlit.
    
    Parametros:
    color_text (str): Nombre del color para los textos generales.
    """
    css_code = f"""
    <style>
    /* Cambiar el color del texto principal */
    body {{
        color: {color_text};
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

def Seleccion_Descartar():   # => list  
    # Seleccionar las variables por descartar
    descartar = st.multiselect(
            'Seleccione las variables a descartar:',
            ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'ninguna'])
    return descartar
    
def Seleccion_Considerar(descartar):   # => list
    # Lista total de variables disponibles
    considerar = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']
 
    # Ciclo para quitar las que descarta el usuario
    for opcion in descartar:
        if opcion == "ninguna" or opcion == None:
            pass
        else:
            considerar.remove(opcion)
    return considerar

def Armado_DataFrame(considerar, y, df):   # => df
    BD_dict = {}
    for i in range(len(considerar)):
        BD_dict[considerar[i]] = df[considerar[i]]
    df1 = pd.DataFrame(BD_dict)
    df1 = pd.concat([df1, y], axis=1)
    return df1

def Seleccion_Datos(v_min, v_max):
    # Seleccionar el número de datos por utilizar:
    num_data = st.slider(
        'Seleccione el número de datos por incluir',
        min_value = v_min,
        max_value = v_max, 
        value = v_max,
        step = 10)
    return num_data

def Seleccion_Unidad_Tiempo():   #=> str
    # Seleccionar la unidad de tiempo
    frec_temp = st.select_slider(
        'Seleccione la unidad de predicción',
        ['dia', 'semana', 'mes', 'año'])
    return frec_temp

def Creacion_Mapa_Calor(X, colores):
    # Crear la matriz de Correlación las variables de entrada
    corrMatrix = X.corr()
    
    # Crear el espacio de la figura con sus dimensiones
    fig1, ax1 = plt.subplots(figsize=(12, 10))

    # Crear el mapa de calor
    sns.heatmap(corrMatrix, annot=True, cmap=colores, fmt=".2f")

    # Título
    ax1.set_title('Matriz de Correlación de las Variables de Entrada')

    # Mostrar el mapa de calor en Streamlit
    st.pyplot(fig1)

def Set_Data_train_test(X, y):
    # División de los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def Ajuste_Modelo(X_train, X_test, y_train, y_test, modelo_deseado):
    # Crear instancia y ajustar el modelo
    if modelo_deseado == 'Lineal':
        model = LinearRegression()
        param_model = model.fit(X_train, y_train)
        return model
    elif modelo_deseado == 'RandomForest':
        model = RandomForestRegressor()
        param_model = model.fit(X_train, y_train)
        return model
    elif modelo_deseado == 'RejillaSCVR':
        model = RandomForestRegressor()
        Estimadores = [100,150,200,250]
        metrica = make_scorer(mean_absolute_error)
        parametros = {'n_estimators': Estimadores, 'criterion': ['absolute_error', 'squared_error']}
        GSCVR = GridSearchCV(model, param_grid = parametros, scoring = metrica, cv = 3, n_jobs = 1)
        GSCVR.fit(X_train, y_train)
        model = GSCVR.best_estimator_
        param_model = model.fit(X_train, y_train)
        return model
    elif modelo_deseado == 'Staking':
        RF_1 = RandomForestRegressor(n_estimators = 200, 
                                     criterion = 'absolute_error',
                                     max_depth = 7, 
                                     min_samples_split = 2, 
                                     min_samples_leaf = 1, 
                                     bootstrap = True, 
                                     n_jobs = 1, 
                                     ccp_alpha = 0.0)

        RF_2 = RandomForestRegressor(n_estimators = 150, 
                                     criterion = 'squared_error', 
                                     max_depth = 10, 
                                     min_samples_split = 2,
                                     min_samples_leaf = 1, 
                                     bootstrap = True, 
                                     n_jobs = 1, 
                                     ccp_alpha = 0.0)

        modelos = [('Random Forest 1', RF_1), ('Random Forest 2', RF_2)]
        LR = LinearRegression()
        model = StackingRegressor(modelos, final_estimator = LR, cv = 2, n_jobs = 1)
        param_model = model.fit(X_train, y_train)
        return model
    elif modelo_deseado == 'XGBBOOST':
        model = XGBRegressor()
        param_model = model.fit(X_train, y_train)
        return model
    elif modelo_deseado == 'LIGHTGBM':
        model = HistGradientBoostingRegressor()
        param_model = model.fit(X_train, y_train)
        return model
    else:
        st.write('**Error al seleccionar el modelo**')
        return None

def Predic_Model(X_train, X_test, model):
    # Predicciones del modelo
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return y_train_pred, y_test_pred 

def Eval_Model(y_train, y_test, y_train_pred, y_test_pred):
    # Evaluación del modelo con la raiz del error cuadrado medio
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    # Evaluación del modelo con r2
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cálculo de los residuos
    residuals = y_test - y_test_pred

    return train_rmse, test_rmse, train_r2, test_r2, residuals 

def Fig_Ajuste_Modelo(y_test, y_test_pred, y_train_pred, X, color, modelo):
    # Crear el espacio de la figura con sus dimensiones
    fig2, ax2 = plt.subplots(figsize=(16, 9))

    # Crear la figura
    plt.scatter(y_test, y_test_pred, label='Predicciones', color=color, alpha=0.3, s=100)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal')

    # Configuración
    ax2.set_title(f'Datos de Entrenamiento y Ajuste (modelo: {modelo}, registros: {len(y_test)}, variables: {len(X.columns)})', fontsize=24)
    ax2.set_xlabel('Valores Reales', fontsize=22)
    ax2.set_ylabel('Predicciones', fontsize=22)
    ax2.set_xlim(0, round(y_train_pred.max()*10 + 0.5, 0)/10)
    ax2.set_ylim(0, round(y_train_pred.max()*10 + 0.5, 0)/10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
  
    # Mostrar la figura en Streamlit
    st.pyplot(fig2)

def Fig_Distrib_Residuos(residuals, X, color, modelo):
    # Crear el espacio de la figura con sus dimensiones
    fig3, ax3 = plt.subplots(figsize=(16, 9))

    # Crear la figura
    sns.histplot(residuals, kde=True, color=color, bins=30, alpha=0.3,)

    # Configuración
    ax3.set_title(f'Distribución de Residuos (modelo: {modelo}, registros: {len(X*0.8)}, variables: {len(X.columns)})', fontsize=24)
    ax3.set_xlabel('Residuos', fontsize=22)
    ax3.set_ylabel('Frecuencia', fontsize=22)
    plt.axvline(x=0, color='red', linestyle='--', label='0')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
  
    # Mostrar la figura en Streamlit
    st.pyplot(fig3)

def Tabla_Resultados_Ajuste(train_rmse, test_rmse, train_r2, test_r2, residuals):
    # Datos
    resultados = {
        "Métrica": ["RMSE de entrenamiento", "RMSE de prueba", "R² de entrenamiento", "R² de prueba", "Residual medio"],
        "Valor": [train_rmse, test_rmse, train_r2, test_r2, residuals.mean()],
    }

    # Crear DataFrame con formateo a 3 decimales
    df_resultados = pd.DataFrame(resultados)
    df_resultados["Valor"] = df_resultados["Valor"].map("{:.4f}".format)

    # Mostrar en Streamlit
    st.dataframe(df_resultados)
    return 

def Fig_Linea_Tiempo_Model(y_test, y_test_pred, modelo):
    # Crear el espacio de la figura con sus dimensiones
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Crear la figura
    plt.plot(np.arange(len(y_test)), y_test, color = 'b', label='Y original')
    plt.plot(np.arange(len(y_test)), y_test_pred, color = 'r', label='Y estimada')
    
    # Configuración
    ax.set_title(f'Precio de Energía Reportado vs Predicción (modelo: {modelo}, registros: {len(y_test)})', fontsize=24)
    ax.set_xlabel('muestras') # Etiqueta del eje x
    ax.set_ylabel('y') # Etiqueta del eje y
    plt.axhline(0, color="black") # Elegir color de la linea horizontal de referencia
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.tight_layout()

    # Mostrar la figura en Streamlit
    st.pyplot(fig)

##################################################################################################
# Utilidades Página Pronóstico
##################################################################################################
def Seleccion_Modelo_Pronostico():
    # Repositorio de Modelos
    folder_path='models/'
    try:
        archivos = [f for f in os.listdir(folder_path) if f.endswith('.sav')]
    except FileNotFoundError:
        st.error(f"La carpeta '{folder_path}' no existe.")
        return None

    # Crear el selector en Streamlit
    modelo_seleccionado = st.selectbox(
        'Seleccione el Modelo de Pronóstico',
        archivos,
        format_func=lambda x: x.split('.')[0] if x != 'Ninguno' else x)
    return modelo_seleccionado

def Seleccion_Nuevos_Datos(original_file="04_datos_sin_outilers_Norm.csv"):
    folder_path='data/nuevos/'
    try:
        archivos = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    except FileNotFoundError:
        st.error(f"La carpeta '{folder_path}' no existe.")
        return None, None

    # Seleccionar archivo de nuevos datos
    new_file = st.selectbox(
        'Seleccione el archivo con los datos para Pronóstico',
        archivos
        )
    if new_file == 'Ninguno':
        return None, None

    # Cargar dataframes
    try:
        df1 = pd.read_csv('data/' + original_file)
        df2 = pd.read_csv(folder_path + new_file)
        
        # Cambiar formato de fecha a datetime y convertir en índice
        df2['Fecha'] = pd.to_datetime(df2['Fecha'], format='%m/%d/%Y')
        df2.set_index('Fecha', inplace=True)
    except:
        st.write('No e pueden cargar los datos')
        return None, None

    return df1, df2

def Normalizar_Datos(df_original, df_nuevo):
    """
    Normaliza las columnas de un dataframe nuevo basado en la escala de la base original.
    """
    columnas = df_original.columns
    df_nuevo_normalizado = df_nuevo.copy()
    for col in columnas:
        if col in df_nuevo.columns:
            min_val = df_original[col].min()
            max_val = df_original[col].max()
            df_nuevo_normalizado[col] = (df_nuevo[col] - min_val) / (max_val - min_val)
    return df_nuevo_normalizado

def Cargar_Modelo_y_Predecir(modelo_file, X_real):
    """
    Carga un modelo desde un archivo .sav y realiza predicciones en los datos de entrada.
    """
    path = 'models/'
    # Cargar el modelo
    with open(path + modelo_file, 'rb') as file:
        modelo = pkl.load(file)

    # Realizar predicciones
    y_pron = modelo.predict(X_real)
    return y_pron







"""
# Ejecutar el modelo para el dato nuevo
def prueba_modelo(arreglo):
    modelo_cargado = load(open('data\modelo_rf.sav', 'rb'))
    prediccion_rf = modelo_cargado.predict(arreglo)
    st.subheader('**Diagnóstico del paciente ingresado')
    st.write('Diagnóstico')
    st.write(f'El paciente ingresado de acuerdo a los datos hallados {prediccion_rf[0]} padece de SMEC')
"""

"""
def Prueba_Modelos():
    GSCVR_Norm_save = open('/content/SCVR_Norm.sav', 'wb')
    dump(Modelo_Elegido, GSCVR_Norm_save)
    Y_pred_Grid_Norm_df = pd.DataFrame(Y_Pred)
    Y_pred_Grid_Norm_df.to_csv('Y_pred_Grid_Norm.csv')
    pass
"""
