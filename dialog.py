# -*-coding:utf-8-*-

from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
from PNJudge import judge
from rnn import exec_attention
from rnn import exex_attention_ver_2

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

@app.route('/dialog/<user_message>', methods=['GET'])
def dialog(user_message):
    print user_message
    response = exec_attention.get_response(user_message, "rnn/attention-models/attention-24.model")
    result = {
            "response": response
            }
    return jsonify(ResultSet=result)

@app.route('/dialog2/<user_message>', methods=['GET'])
def dialog(user_message):
    print user_message
    response = exec_attention.get_response(user_message, "rnn/attention-models/attention-99.model")
    result = {
            "response": response
            }
    return jsonify(ResultSet=result)

if __name__ == "__main__":
    app.run(debug=True)
