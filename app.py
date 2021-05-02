from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("Students_mark_predictor_model.pkl")

df = pd.DataFrame()

@app.route('/')
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    global df

    input_features = [float(x) for x in request.form.values()]
    value = np.array(input_features)

    #validate input hours    
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
    
    output = model.predict([value])[0][0].round(2)
    # input and predicted value store in df then save in csv file
    df= pd.concat([df,pd.DataFrame({'Study Hours':input_features,'Predicted Output':[output]})],ignore_index=True)
    print(df)   
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html', Prediction_text = f"you will get {output}% marks, when you do study {input_features} hours per day")

if __name__ == '__main__':
    app.debug = True
    app.run()