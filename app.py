from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained ML model (make sure model.pkl is in the same directory)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')  # Make sure index.html is in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cgpa = float(request.form['cgpa'])
        iq = int(request.form['iq'])

        input_data = np.array([[cgpa, iq]])
        prediction = model.predict(input_data)[0]

        result = "✅ Likely to be Placed" if prediction == 1 else "❌ Unlikely to be Placed"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)

