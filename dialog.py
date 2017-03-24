# -*-coding:utf-8-*-

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PNJudge import judge

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World!"

@app.route('/responce')
def responce():
    return "テスト返答です"

@app.route('/pn_judge/<sentense>', methods=['GET'])
def pn_judge(sentense):
    sentense = sentense.encode("utf-8")
    value = judge.judge_pn(sentense)
    result = "ポジティブ" if value > 0 else "ネガティブ"
    return result

if __name__ == "__main__":
    app.run()
