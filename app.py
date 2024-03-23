from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model_iris = pickle.load(open('savedmodel.sav', 'rb'))
model_titanic = pickle.load(open('titanic_dt.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', **locals())


@app.route('/predict_iris', methods=['POST', 'GET'])
def predict_iris():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        result = model_iris.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    return render_template('iris.html', **locals())


def map_sex(sex):
    return 0 if sex.lower() == 'male' else 1

@app.route('/predict_titanic', methods=['POST', 'GET'])
def predict_titanic():
    if request.method == 'POST':
        Pclass = float(request.form['Pclass'])
        Sex = map_sex(request.form['Sex'])
        Age = int(request.form['Age'])
        SibSp = float(request.form['SibSp'])
        result = model_titanic.predict([[Pclass, Sex, Age, SibSp]])[0]
        result = 'Survaved' if result == 0 else 'Death'
    return render_template('titanic.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)
