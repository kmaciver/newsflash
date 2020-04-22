from application import app
from flask import render_template, request, jsonify, json
import numpy as np
import application.TextGeneratorBusiness

@app.route("/")
def home():
    return '<h1>Backend Working from Docker</h1>'


#initiate class
Generator = application.TextGeneratorBusiness.TextGenerator()

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route('/read_text', methods=['GET','POST'])
def readText():
    typedText = request.get_data()
    if len(typedText.split()) < 5:
        # print('connected') # Debugging
        return(typedText)
    else:
        # print('started') # Debugging
        Generator.text = typedText.decode('ASCII') 
        candidates = Generator.generatecandidates()
        # print('ok up to here') # Debugging
        return (json.dumps(candidates, cls=NumpyEncoder))
