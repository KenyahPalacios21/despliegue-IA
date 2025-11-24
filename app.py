from flask import Flask, render_template, request
import numpy as np
import json
import os
import joblib

app = Flask(__name__)

modelo = joblib.load("models/modelo_regresion_logistica.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("json/modelo_regresion_logistica_info.json", "r") as f:
    modelo_info = json.load(f)

features = modelo_info["features"]

traducciones = {
    "Age": "Edad",
    "Sex": "Sexo (1=Hombre, 2=Mujer)",
    "Estado_Civil": "Estado civil (0/1)",
    "Ciudad": "Ciudad (0/1)",
    "Steroid": "Esteroides (0/1)",
    "Antivirals": "Antivirales (0/1)",
    "Fatigue": "Fatiga (0/1)",
    "Malaise": "Malestar general (0/1)",
    "Anorexia": "Anorexia (0/1)",
    "Liver_Big": "Hígado agrandado (0/1)",
    "Liver_Firm": "Hígado endurecido (0/1)",
    "Spleen_Palpable": "Bazo palpable (0/1)",
    "Spiders": "Arañas vasculares (0/1)",
    "Ascites": "Ascitis (0/1)",
    "Varices": "Várices (0/1)",
    "Bilirubin": "Bilirrubina",
    "Alk_Phosphate": "Fosfatasa alcalina",
    "Sgot": "SGOT",
    "Albumin": "Albúmina",
    "Protime": "Tiempo de protrombina",
    "Histology": "Histología (0/1)"
}

@app.route("/")
def index():
    return render_template("index.html", features=features, traducciones=traducciones)

@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        valores = []

        for feature in features:
            valor = float(request.form.get(feature))
            valores.append(valor)

        valores_array = np.array([valores])
        valores_escalados = scaler.transform(valores_array)

        pred = modelo.predict(valores_escalados)[0]
        prob = modelo.predict_proba(valores_escalados)[0][1]

        return render_template(
            "index.html",
            features=features,
            traducciones=traducciones,
            resultado=pred,
            prob=round(prob * 100, 2)
        )

    except Exception as e:
        return f"Error procesando la predicción: {e}"

if __name__ == "__main__":
    app.run()
