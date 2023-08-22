from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            gender = int(request.form.get('gender')),
            age = float(request.form.get('age')),
            annual_salary = float(request.form.get('annual_salary')),
            credit_card_debt = float(request.form.get('credit_card_debt')),
            net_worth = int(request.form.get('net_worth'))
        )

        new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(new_data)

        return render_template('result.html',final_result = pred)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)