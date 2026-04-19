from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    features = list(data.values())

    prediction = model.predict([features])

    return jsonify({
        "Predicted Marks": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)