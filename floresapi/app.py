from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pickle
import numpy as np

#Cargamos nuestros modelos
with open('models/model.pkl','rb') as f:
    model = pickle.load(f)

#carfamos el archivo para estandarizar
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Creamos la API
app = FastAPI()

# Crear enlace a la carpeta static o montamos la carpeta static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Creamos el template qeu se va aejcutar cuando levantemos el servicio y entremos al backend
templates = Jinja2Templates(directory="templates")


# Creo el endpoints para el consumo de la API
# Formulario enlace
@app.get("/", response_class=HTMLResponse)
# Creamos una función para mostrar el index
def formulario(request: Request):
    return templates.TemplateResponse(
        request = request,
        name = "index.html",
        context = {"request": request, "resultado": None, "mensaje": None}
    )

#Implementamos una funcion para l predicción
@app.post("/predict", response_class=HTMLResponse)
#Creo la funcion para obtener los parametro del formulario
def predecir(request: Request,
             atributo1: float = Form(...),
             atributo2: float = Form(...),
             atributo3: float = Form(...),
             atributo4: float = Form(...),
             ):
    #convertimos los datos a un arreglo unidimensional
    datos = np.array([[atributo1,atributo2,atributo3,atributo4]])
    #escalamos los nuevos datos
    datos_estandarizados = scaler.transform(datos)
    #realizamos la prediccion del modelo de aprendizaje
    prediccion = model.predict(datos_estandarizados)
    #mostramos el resultado de la prediccion
    resultado = int(prediccion[0])
    #mostramos los mensajes indicados
    mensaje=""
    #hago condicion para verificar el resultado de la prediccion
    if resultado == 0:
        mensaje="La prediccion con los datos ingresados corresponde a la flor setosa"
    elif resultado == 1:
        mensaje="La prediccion con los atos ingresados corresponde a la flor Versicolor"
    else:
        mensaje="La prediccion con los datos ingresados corresponde a la flor Virginica"
    #retornamos los resultados
    return templates.TemplateResponse(
        request = request,
        name="index.html",
        context={"request": request, "resultado": resultado, "mensaje": mensaje},
                                      )