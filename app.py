from flask import Flask, request, jsonify
import joblib
import numpy as np

#Definir el puerto para el servidor
port = 3010;

# Cargar el modelo previamente guardado
model = joblib.load("hair_loss_model_v2.bin")

# Crear la aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el cuerpo de la solicitud
        data = request.json

        # Validar que todos los features necesarios estén en el cuerpo de la solicitud
        features = ['genetica', 'cambios_hormonales', 'condiciones_medicas',
    'medicamentos_tratamientos', 'deficiencias_nutricionales', 'estres', 'edad',
    'malos_habitos_cuidado_capilar', 'factores_ambientales', 'tabaquismo',
    'perdida_peso']

        if not all(feature in data for feature in features):
            return jsonify({'error': 'Missing one or more required features'}), 400

        # Convertir los datos en un array numpy para el modelo
        input_data = np.array([[data[feature] for feature in features]])

        # Realizar la predicción
        prediction = model.predict(input_data)

        # Retornar el resultado
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Iniciar la aplicación
if __name__ == '__main__':
    #prod
    app.run(host='0.0.0.0', port=port)
    #debug
    #app.run(host='0.0.0.0', port=port, debug=True)
