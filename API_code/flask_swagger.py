import pandas as pd
from flask import Flask, request, jsonify, make_response, Response, render_template
from flask_swagger_ui import get_swaggerui_blueprint
from sentimentprediction_NN import sentiment_text, sentiment_file
import joblib

# Init app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False 

# flask swagger configs
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Sentiment"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)


# Homepage
@app.route('/', methods=['GET'])
def get():
    return "WELCOME" 


@app.route('/RNN/text', methods=["POST"])
def RNN_text():
    if request.method == "POST":
        input_text = str(request.form["text"])
        sentiment = sentiment_text(input_text,'RNN')
        output = dict(input=input_text, sentiment=sentiment)
        return jsonify(output)

@app.route('/RNN/file', methods=["POST"])
def RNN_file():
    if request.method == "POST":
        file = request.files['file']

        try:
            file = pd.read_csv(file, encoding='iso-8859-1')
        except:
            try:
                file = pd.read_csv(file, encoding='utf-8')
            except:
                try:
                    file = pd.read_csv(file, sep='\t')
                except:
                    pass
        print("======== read data csv to pandas =========")
        print(type(file))
        if(isinstance(file, pd.DataFrame)):
            file = sentiment_file(file,'RNN')
            if(isinstance(file, pd.DataFrame)):
                response = Response(file.to_json(orient="records"), mimetype='application/json')
            else:
                response = "Error"
        else:
            response = "Error"
        return response





@app.route('/CNN/text', methods=["POST"])
def CNN_text():
    if request.method == "POST":
        input_text = str(request.form["text"])
        sentiment = sentiment_text(input_text,'CNN')
        output = dict(input=input_text, sentiment=sentiment)
        return jsonify(output)

@app.route('/CNN/file', methods=["POST"])
def CNN_file():
    if request.method == "POST":
        file = request.files['file']

        try:
            file = pd.read_csv(file, encoding='iso-8859-1')
        except:
            try:
                file = pd.read_csv(file, encoding='utf-8')
            except:
                try:
                    file = pd.read_csv(file, sep='\t')
                except:
                    pass
        print("======== read data csv to pandas =========")
        print(type(file))
        if(isinstance(file, pd.DataFrame)):
            file = sentiment_file(file,'CNN')
            if(isinstance(file, pd.DataFrame)):
                response = Response(file.to_json(orient="records"), mimetype='application/json')
            else:
                response = "Error"
        else:
            response = "Error"
        return response





@app.route('/LSTM/text', methods=["POST"])
def LSTM_text():
    if request.method == "POST":
        input_text = str(request.form["text"])
        sentiment = sentiment_text(input_text,'lstm')
        output = dict(input=input_text, sentiment=sentiment)
        return jsonify(output)

@app.route('/LSTM/file', methods=["POST"])
def LSTM_file():
    if request.method == "POST":
        file = request.files['file']

        try:
            file = pd.read_csv(file, encoding='iso-8859-1')
        except:
            try:
                file = pd.read_csv(file, encoding='utf-8')
            except:
                try:
                    file = pd.read_csv(file, sep='\t')
                except:
                    pass
        print("======== read data csv to pandas =========")
        print(type(file))
        if(isinstance(file, pd.DataFrame)):
            file = sentiment_file(file,'LSTM')
            if(isinstance(file, pd.DataFrame)):
                response = Response(file.to_json(orient="records"), mimetype='application/json')
            else:
                response = "Error"
        else:
            response = "Error"
        return response
    



@app.route('/NN/text', methods=["POST"])
def mlp_model_text():
    if request.method == "POST":
        input_text = str(request.form["text"])
        sentiment = sentiment_text(input_text,'mlp_model')
        
        output = dict(input=input_text, sentiment=sentiment)
        return jsonify(output)

@app.route('/NN/file', methods=["POST"])
def mlp_model_file():
    if request.method == "POST":
        file = request.files['file']

        try:
            file = pd.read_csv(file, encoding='iso-8859-1')
        except:
            try:
                file = pd.read_csv(file, encoding='utf-8')
            except:
                try:
                    file = pd.read_csv(file, sep='\t')
                except:
                    pass
        print("======== read data csv to pandas =========")
        print(type(file))
        if(isinstance(file, pd.DataFrame)):
            file = sentiment_file(file,'mlp_model')
            if(isinstance(file, pd.DataFrame)):
                response = Response(file.to_json(orient="records"), mimetype='application/json')
            else:
                response = "Error"
        else:
            response = "Error"
        return response



# error handling
@app.errorhandler(400)
def handle_400_error(_error):
    "Return a http 400 error to client"
    return make_response(jsonify({'error': 'Misunderstood'}), 400)


@app.errorhandler(401)
def handle_401_error(_error):
    "Return a http 401 error to client"
    return make_response(jsonify({'error': 'Unauthorised'}), 401)


@app.errorhandler(404)
def handle_404_error(_error):
    "Return a http 404 error to client"
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(500)
def handle_500_error(_error):
    "Return a http 500 error to client"
    return make_response(jsonify({'error': 'Server error'}), 500)


#Run Server
if __name__ == '__main__':
    app.run(debug=True)
# Default IP 127.0.0.1 Port 5000








