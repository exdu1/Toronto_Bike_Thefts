from flask import Flask, request, jsonify
import numpy as np
import joblib  # Import joblib to load the saved model

app = Flask(__name__)

# Load the saved model and scaler
print("Loading model...")
model = joblib.load('model.joblib')
print("Model loaded successfully.")

scaler = joblib.load('scaler.joblib')  # Load the scaler

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Request received")  # Log when request is received

        # Get JSON data from the POST request
        data = request.get_json()
        print("Data received:", data)  # Log the received data

        # Extract features as you normally would
        features = [
            data["OBJECTID"], data["EVENT_UNIQUE_ID"], data["PRIMARY_OFFENCE"],
            data["OCC_DATE"], data["OCC_YEAR"], data["OCC_MONTH"], data["OCC_DOW"],
            data["OCC_DAY"], data["OCC_DOY"], data["OCC_HOUR"], data["REPORT_DATE"],
            data["REPORT_YEAR"], data["REPORT_MONTH"], data["REPORT_DOW"],
            data["REPORT_DAY"], data["REPORT_DOY"], data["REPORT_HOUR"], data["DIVISION"],
            data["LOCATION_TYPE"], data["PREMISES_TYPE"], data["BIKE_MAKE"],
            data["BIKE_MODEL"], data["BIKE_TYPE"], data["BIKE_SPEED"], data["BIKE_COLOUR"],
            data["BIKE_COST"], data["HOOD_158"], data["NEIGHBOURHOOD_158"], data["HOOD_140"],
            data["NEIGHBOURHOOD_140"], data["LONG_WGS84"], data["LAT_WGS84"], data["x"], data["y"]
        ]

        # Convert features to a numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Scale the features
        print("Features before scaling:", features_array)
        features_scaled = scaler.transform(features_array)
        print("Features after scaling:", features_scaled)

        # Get prediction from the model
        prediction = model.predict(features_scaled)

        # Log the prediction
        print("Prediction:", prediction)

        # Return the prediction in JSON format
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Set the port to 5001
