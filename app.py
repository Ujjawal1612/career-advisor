from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model and encoders
with open('career_model.pkl', 'rb') as file:
    model, le_stream, le_interest, le_career = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# ✅ REPLACE THE OLD PREDICT FUNCTION WITH THIS:
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        stream = data.get('stream')
        interest = data.get('interest')

        # Check if input values exist in trained LabelEncoders
        if stream not in le_stream.classes_:
            return jsonify({'error': f"Stream '{stream}' is not in the trained model."})
        if interest not in le_interest.classes_:
            return jsonify({'error': f"Interest '{interest}' is not in the trained model."})

        # Convert inputs to numerical values
        stream_encoded = le_stream.transform([stream])[0]
        interest_encoded = le_interest.transform([interest])[0]

        # Predict career
        prediction = model.predict(np.array([[stream_encoded, interest_encoded]]))
        career = le_career.inverse_transform(prediction)[0]

        return jsonify({'career': career})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
