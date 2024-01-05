import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import numpy as np

# -------------------------PROCESO DE DESPLIEGUE------------------------------
# Correr despliegue_py.ipynb, y generar el Modelo1.joblib, en Jupyter.
# El archivo Modelo1.joblib fue hecho con: sklearn 1.3.2
# El origen del Modelo1.joblib puede generar conflicto.
#En consola:
# pip install scikit-learn==1.3.2

#01 --------------------------Load the model-------------------------------------------
clf = load('Modeloestandarizado_pipeline.joblib')

#02---------------- Variables globales para los campos del formulario-----------------------
pregnancies = 0.0
glucose = 0.0
blood_pressure = 0.0
skin_thickness = 0.0
insulin = 0.0
bmi = 0.0
diabetes_pedigree_function = 0.0
age = 0.0

#03 Reseteo------------- Flag to track error---------------------------------------
error_flag = False

# Reset inputs function
def reset_inputs():
    global pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, error_flag
    pregnancies = glucose = blood_pressure = skin_thickness = insulin = bmi = diabetes_pedigree_function = age = 0.0
    error_flag = False

# Inicializar variables
reset_inputs()
#-----------------------------------------------------------------------------------------------


# ------------------------Título centrado-------------------------------------------------
st.title("Modelo Predictivo de Diabetes con Decision Tree Classifier")
st.markdown("Este conjunto de datos es originalmente del Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales. El objetivo es predecir en base a mediciones de diagnóstico si un paciente tiene diabetes.Se colocaron varias restricciones en la selección de estas instancias de una base de datos más grande. En particular, todos los pacientes aquí son mujeres de al menos 21 años de herencia indígena Pima.")
st.markdown("---")

#----------------------- Función para validar los campos del formulario----------------------------
def validate_inputs():
    global error_flag
    if any(val < 0 for val in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]):
        st.error("No se permiten valores negativos. Por favor, ingrese valores válidos en todos los campos.")
        error_flag = True
    else:
        error_flag = False

#------------------------------------ Formulario en dos columnas------------------------------------
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)

    # Input fields en la primera columna
    with col1:
        pregnancies = st.number_input("**Embarazos(Min.=1, Max.=17)**", 
                                      min_value=0.0, 
                                      value=pregnancies, 
                                      step=4.0, 
                                      format="%f",
                                      help="Número de embarazos")
        glucose = st.number_input("**Glucosa (Min.=44, Max.=199)**", 
                                  min_value=0.0, 
                                  value=float(glucose), 
                                  step=1.0, format="%f",
                                  help="Nivel de glucosa en plasma a 2 horas en una prueba oral de tolerancia a la glucosa")
        blood_pressure = st.number_input("**Presión Sanguínea (Min.=24, Max.=122)**", 
                                         min_value=0.0, 
                                         value=blood_pressure, 
                                         step=1.0,
                                         format="%f", help="Presión arterial diastólica (mm Hg)")
        skin_thickness = st.number_input("**Espesor de la Piel (Min.=7, Max.=99)**",
                                         min_value=0.0, 
                                         value=skin_thickness,
                                         step=1.0,
                                         format="%f",
                                         help="Espesor del pliegue cutáneo del tríceps (mm)")

    # Input fields en la segunda columna
    with col2:
        insulin = st.number_input("**Insulina (Min.=14, Max.=846)**", 
                                  min_value=0.0,
                                  value=insulin,
                                  step=1.0, 
                                  format="%f",
                                  help="Insulina sérica en mu U/ml")
        bmi = st.number_input("**IMC (Min.=18.2, Max.=67.1)**", 
                              min_value=0.0, 
                              value=bmi, 
                              step=1.0, 
                              format="%f",
                              help="Índice de masa corporal (peso en kg / (altura en m)^2)")
        diabetes_pedigree_function = st.number_input("***Función de Pedigrí de la Diabetes(Min.=0.078, Max.=2.42)**",
                                                     min_value=0.0,
                                                     value=diabetes_pedigree_function,
                                                     step=0.01,
                                                     format="%f", 
                                                     help="Función de pedigree de diabetes")
        age = st.number_input("**Edad(Min.=21, Max.=81)**", 
                                                     min_value=0, 
                                                     step=1, 
                                                     format="%d",
                                                     help="Edad")

#----------------------------------------- Boton de Predecir-------------------------------------------------
    predict_button = st.form_submit_button("Predecir", on_click=validate_inputs, args=())

# Validar que no haya valores negativos en los campos cuando se presiona el botón
# Si hay error no permita seguir tipeando!!!!!!!!!!!!!!!!!!!
if predict_button and error_flag:
    st.stop()

if predict_button and not error_flag:
    # Crear DataFrame
    data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    }
    df = pd.DataFrame(data)

    # ----------------------ESCALANDO LOS DATOS----------------------------------------
    ## Todo este proceso lo hace el pipeline del modelo!!!!!!!!!!!!!!!!!!!!!!
    #numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    #scaler = StandardScaler()
    # means es el vector de medias de la base de datos original, para cuantitativas
    #means = pd.DataFrame([4.38215017, 121.64296972,  72.36209693,  28.9114078 ,
    #   152.40941338,  32.4456522 ,   0.4718763 ,  33.24088542])
    #std_devs = pd.DataFrame([3.03935172, 30.46669662, 12.14734621,  9.51849562, 97.51826456,
    #    6.87872922,  0.3313286 , 11.76023154])

    # Utilizar .values.flatten() para obtener un array unidimensional
    #scaler.mean_ = means.values.flatten()
    #scaler.scale_ = std_devs.values.flatten()
    #df[numeric_cols] = scaler.transform(df[numeric_cols])
    #df = pd.DataFrame(df, columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
    #                                    'DiabetesPedigreeFunction','Age'])
    #---------------------------------------------------------------------------------------------------
if predict_button and error_flag:
    st.stop()

if predict_button and not error_flag:
    # ... (rest of the code)

    # Realizar predicción
    probabilidades_clases = clf.predict_proba(df)[0]
    import numpy as np

    # Obtener la clase con la mayor probabilidad
    clase_predicha = np.argmax(probabilidades_clases)

    # Asignar salida y probabilidad según la clase predicha
    # En el script original: #Outcome: 0 No es diabético;  1 es diabético
    if clase_predicha == 0:
        salida = "NO Diabetes"
        probabilidad_diabetes = probabilidades_clases[0]
        estilo_resultado = 'background-color: lightgreen; font-size: larger;'
    else:
        salida = "DIABETES"
        probabilidad_diabetes = probabilidades_clases[1]
        estilo_resultado = 'background-color: lightcoral; font-size: larger;'

    # Mostrar resultado con estilo personalizado
    resultado_html = f"<div style='{estilo_resultado}'>La predicción fue de clase '{salida}' con una probabilidad de {round(float(probabilidad_diabetes), 4)}</div>"
    st.markdown(resultado_html, unsafe_allow_html=True)

# --------------------------- Boton de Resetear-------------------------------------

if st.button("Resetear. Para resetear el formulario presiona F5"):
    # Resetear inputs
    reset_inputs()
    # Recargar la página usando JavaScript (solución alternativa)
    st.markdown("<script type='text/javascript'>window.location.reload();</script>", unsafe_allow_html=True)
  


