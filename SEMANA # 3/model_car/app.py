from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from flask_cors import CORS
import prediction

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

class GetPrediction(Resource):
    def get(self):
        return {"error": "Invalid Method."}, 405

    def post(self):
        try:
            data = request.get_json()
            predict = prediction.predict_price(data)
            return jsonify({"Precio": int(predict)})

        except Exception as error:
            return {"error": str(error)}, 500

api.add_resource(GetPrediction, '/getPrediction')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
