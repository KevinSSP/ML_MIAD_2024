from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from flask_cors import CORS
import utils
import predictionv2
import json

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

class GetPrediction(Resource):
    def get(self):
        return {"error": "Invalid Method."}, 405

    def post(self):
        try:
            data = request.get_json()
            predict = predictionv2.main(data)
            # Convertir la cadena JSON en un diccionario Python
            result_list = json.loads(predict)

            # Extraer el primer elemento de la lista (el único en este caso)
            result_dict = result_list[0]

            # Crear la respuesta formateada
            formatted_response = {"result": result_dict}

            return formatted_response

        except Exception as error:
            return {"error": str(error)}, 500

api.add_resource(GetPrediction, '/getPrediction')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
