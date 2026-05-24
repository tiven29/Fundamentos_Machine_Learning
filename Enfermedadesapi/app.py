# Importamos las librerias a ocupar
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pickle
import numpy as np

# cargamos nuestros modelos
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Cargamos el archivo para estandarizar
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Creamos la API
app = FastAPI()

# Creamos el enlace a nuestra carpeta static
app.mount("/static", StaticFiles(directory="static"), name="static")

# creamos el template el cual se va a ejecuar al momento  de levantar el servicio
templates = Jinja2Templates(directory="templates")


# Creamos el endpoints para el consumo de la API
# Formulario enlace
@app.get("/", response_class=HTMLResponse)
# Creamos una funcion para mostrar el index
def formulario(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "resultado": None, "mensaje": None},
    )


# Implementamos una funcion para la prediccion
@app.post("/predict", response_class=HTMLResponse)
# Creamos la funcion para obtener los paraetros del formulario
def predecir(
    request: Request,
    age: float = Form(...),
    sex: float = Form(...),
    cp: float = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: float = Form(...),
    restecg: float = Form(...),
    thalach: float = Form(...),
    exang: float = Form(...),
    oldpeak: float = Form(...),
    slope: float = Form(...),
    ca: float = Form(...),
    thal: float = Form(...),
):
    # convertimos los datos a un arreglo unidimensional
    datos = np.array(
        [
            [
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]
        ]
    )
    # Escalamos los nuevos datos
    datos_estandarizados = scaler.transform(datos)
    # Realizamos la prediccion del modelo de aprendizaje
    prediccion = model.predict(datos_estandarizados)
    # Mostramos el resultado de la prediccion
    resultado = int(prediccion[0])
    # Mostramos los mensajes indicados
    mensaje = ""
    # Realizamos condicion para verificar el resultado de la prediccion
    if resultado == 1:
        mensaje = "El paciente en base a los datos ingresados está SANO (Sin enfermedad cardíaca)."
    elif resultado == 0:
        mensaje = "El paciente en base a los datos ingresados está ENFERMO (Presencia de enfermedad cardíaca)."
    # retornamos los resultados
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "resultado": resultado, "mensaje": mensaje,},
    )
