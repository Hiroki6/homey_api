# -*-coding:utf-8-*-

from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
from PNJudge import judge
from rnn import exec_attention

app = Flask(__name__)

@app.route('/')
def index():
    result = {
            "response": "Hello World!"
            }
    return jsonify(ResultSet=result)

@app.route('/responce')
def responce():
    return "テスト返答です"

@app.route('/pn_judge/<sentense>', methods=['GET'])
def pn_judge(sentense):
    sentense = sentense.encode("utf-8")
    value = judge.judge_pn(sentense)
    result = "ポジティブ" if value > 0 else "ネガティブ"
    return result

@app.route('/dialog/', methods=['POST'])
def dialog():
    user_message = request.form['user_message']
    response = exec_attention.get_response(user_message, "rnn/attention-models/attention-24.model")
    result = {
            "response": response
            }
    return jsonify(ResultSet=result)

if __name__ == "__main__":
    app.run()
