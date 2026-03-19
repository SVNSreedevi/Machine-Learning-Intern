from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


def score_category(score):
    if score >= 85:
        return 'Excellent'
    elif score >= 70:
        return 'Good'
    elif score >= 50:
        return 'Average'
    else:
        return 'Needs Improvement'


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['attendance']),
                float(request.form['study_hours']),
                float(request.form['internal_marks']),
                float(request.form['previous_percentage']),
                float(request.form['assignments_completed']),
                float(request.form['class_test_score'])
            ]

            arr = np.array(features).reshape(1, -1)
            arr_scaled = scaler.transform(arr)

            pred = model.predict(arr_scaled)[0]
            pred = round(float(pred), 1)

            result = {
                'score': pred,
                'category': score_category(pred)
            }

        except Exception as e:
            result = {'error': str(e)}

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
