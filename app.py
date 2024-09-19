from flask import Flask, render_template, request, flash, url_for, redirect, session, make_response
from datetime import datetime

app = Flask(__name__)

#毎回エラー表示は不安なので訪れた時刻でも表示しておく
@app.route("/")
def saisyo():
    now = datetime.now().replace(microsecond=0)
    return f"<h2>このページに訪れた時刻は{now}<br>上のURLバーを正しく書き換えて。</h2>"