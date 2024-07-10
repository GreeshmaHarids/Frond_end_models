from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__,template_folder='template')

model=pickle.load(open("models/model_d.pkl",'rb'))
vec = pickle.load(open("models/vec.pkl", 'rb'))  # Assuming vectorizer is saved as vectorizer.pkl


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['text']
    transformed_text=vec.transform([text])

    prediction=model.predict(transformed_text)

    if prediction[0] == 1:
        output = "Non Spam"

    else:
        output = "Spam"
    # print("Final output:", prediction[0])


    return render_template('index.html', prediction_text='Given text is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)