from flask import Flask, render_template, request
from PROJECT.project__OOP import Stock_Data, Stock_Predict, Trend

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        companyname = request.form['companyname']
        num_days = int(request.form['num_days'])
        quantity = int(request.form['quantity'])
        unit = request.form['unit']

        company = Stock_Data(companyname, (quantity, unit))
        # print("Fetching {} {} data for {}".format(company.unit, company.quantity, company.get_company()))
        df = company.get_stock_data()

        if request.form['submit'] == "Model":
            ##### Prediction ####
            predictor = Stock_Predict(df, num_days)
            graph = predictor.prophet_model()
            future = predictor.future
            current = predictor.ending_stock_price
            next = predictor.next_price
            suggestion = predictor.recommend()
            return render_template('result.html', graph=graph, tables=[future.to_html()],
                                   current=current, next=next, suggestion=suggestion)

        elif request.form['submit'] == "Trend":
            trend = Trend(df)
            graph1, graph2, graph3 = trend.graph_analysis()

            return render_template("visualize.html", graph1=graph1, graph2= graph2, graph3= graph3)

        else:
            return render_template("404.html")


if __name__ == '__main__':
    app.run()

