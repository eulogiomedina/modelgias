from flask import Flask, request, jsonify
from flask_cors import CORS          # <--- Importa flask_cors
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)   # <--- Habilita CORS para cualquier origen (útil para pruebas con React/Render)

# Cargar modelo entrenado
modelo = joblib.load('modelo_randomforest_gias.pkl')

# Variables esperadas por el modelo
columnas = [
    'nivel_compromiso_financiero',
    'tiene_ingreso_fijo',
    'puntual_en_ahorros_previos',
    'ingreso_mensual_aprox',
    'cuenta_con_ahorros',
    'tiene_dependientes',
    'egresos_mensuales_aprox'
]

@app.route('/api/prediccion', methods=['POST'])
def predecir():
    datos = request.get_json()

    try:
        entrada = [float(datos[col]) for col in columnas]
    except KeyError as e:
        return jsonify({"error": f"Falta la variable: {str(e)}"}), 400
    except ValueError:
        return jsonify({"error": "Todas las variables deben ser numéricas"}), 400

    X = np.array(entrada).reshape(1, -1)

    prediccion = modelo.predict(X)[0]
    probabilidad = modelo.predict_proba(X)[0][int(prediccion)]

    return jsonify({
        'es_buen_pagador': int(prediccion),
        'probabilidad': round(probabilidad, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
