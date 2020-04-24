import requests
from application import app
from flask import render_template, request, jsonify, json

@app.route("/")
def home():
    return render_template('home.html')


@app.route('/read_text', methods=['GET','POST'])
def readText():
    typedText = request.get_data()
    print(typedText)
    url = "http://127.0.0.1:8080/read_text"

    payload = typedText
    headers = {
    'Content-Type': 'text/plain'
    }

    response = requests.request("POST", url, headers=headers, data = payload)

    return(response.text.encode('utf8'))