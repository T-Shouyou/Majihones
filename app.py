import cv2
import numpy as np
import pickle
import boto3
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import sqlite3
import os
import secrets
import random
import string
import re
import requests
from datetime import datetime, timedelta
from werkzeug.utils import escape
# デバッグ用のログ出力
import logging


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

app.secret_key = secrets.token_hex(16)

# Google Gemini APIのエンドポイントとAPIキー
# 本番では検索して消せ、WSGIにでも書いて
API_KEY = 'AIzaSyBneMFN_MwtaCXfQih0dqikuTYQKKCmU-s' 
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'

# 自分のパソコンで実行する際の画像の保存先のパス、本番では検索して消せ
LOCAL_IMAGE_FOLDER = 'static/hiroba_img'
# LOCAL_IMAGE_FOLDER = os.path.join(app.root_path, 'static', 'hiroba_img')

# SQLiteデータベースの設定
# 検索してナンバーを消せ
# DATABASE = '/home/UminekoSakana/mysite/mydatabase.db'  # パスが正しいか確認
DATABASE = 'mydatabase.db'  # SQLiteデータベースのファイル名

def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn

# S3クライアントの初期化
s3_client = boto3.client('s3', region_name='us-east-1')  # リージョンを指定

# 事前に計算した料理の特徴をロード
# 検索してナンバーを消せ
# with open('/home/UminekoSakana/mysite/recipe_features.pkl', 'rb') as f:
#     recipe_features = pickle.load(f)
with open('recipe_features.pkl', 'rb') as f:
    recipe_features = pickle.load(f)

def process_image_from_s3(image_path):
    # S3から画像を取得
    img_data = s3_client.get_object(Bucket='gazou', Key=image_path)['Body'].read()
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 画像を処理
    img = cv2.resize(img, (150, 150))
    histogram = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram

def average_features(image_paths):
    features_list = [process_image_from_s3(path) for path in image_paths]
    return np.mean(features_list, axis=0)

def identify_dishes_from_multiple_images(image_paths):
    features = average_features(image_paths)  # 複数の画像の特徴を平均化
    distances = {}

    # 既知の料理特徴と比較
    for dish, recipe_hist in recipe_features.items():
        distance = np.linalg.norm(features - recipe_hist)
        distances[dish] = distance

    # 距離を基に料理名をソートして上位3つを取得
    sorted_dishes = sorted(distances, key=distances.get)[:3]
    return sorted_dishes  # 上位3つの料理名を返す

# def hiroba_img(image_file):

#     image_path = f"hiroba_img/{image_file.filename}"  
    
#     s3_client.upload_fileobj(image_file, 'gazou', image_path)
    
#     img_data = s3_client.get_object(Bucket='gazou', Key=image_path)['Body'].read()
#     img_array = np.frombuffer(img_data, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#     img = cv2.resize(img, (150, 150))
#     histogram = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     cv2.normalize(histogram, histogram)

#     return histogram



@app.route('/')
def index():
    return render_template('top/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('image')  # 複数の画像を取得
    image_paths = []

    # S3に画像をアップロードし、パスを保存
    for file in files:
        image_path = f"uploads/{file.filename}"
        s3_client.upload_fileobj(file, 'gazou', image_path)  # S3にアップロード
        image_paths.append(image_path)  # パスをリストに追加

    # それぞれの認識方法を呼び出し
    recognized_from_multiple = identify_dishes_from_multiple_images(image_paths)
    recognized_by_average = identify_dishes_from_multiple_images(image_paths)  # 平均特徴による認識

    # 上位3つの料理名をまとめる
    predicted_labels = (recognized_from_multiple, recognized_by_average)  # タプルとしてまとめる

    return render_template('ninnsiki/success.html', predicted_labels=predicted_labels)  # 成功画面に遷移

@app.route('/ninnsiki/upload_recipe', methods=['POST'])
def upload_recipe():
    label = request.form['label']  # 入力されたラベルを取得
    file = request.files['image']
    image_path = f"recipe_images/{file.filename}"  # S3のrecipe_imagesフォルダに保存するパス

    # S3に画像をアップロード
    s3_client.upload_fileobj(file, 'gazou', image_path)  # 'gazou'はバケット名

    # extract_features.pyにラベルと画像パスを追加する処理を呼び出す
    update_recipe_features(label, image_path)



    return render_template('ninnsiki/recipe_add_success.html')  # 成功画面に遷移

def update_recipe_features(label, image_path):
    # 既存の料理特徴を読み込む
    with open('recipe_features.pkl', 'rb') as f:
        recipe_features = pickle.load(f)

    # 新しい画像の特徴を計算
    histogram = process_image_from_s3(image_path)

    # ラベルに対して特徴を追加
    if label in recipe_features:
        recipe_features[label].append(histogram)  # 既存のリストに追加
    else:
        recipe_features[label] = [histogram]  # 新規ラベルの場合リストを作成

    # 特徴をファイルに保存
    with open('recipe_features.pkl', 'wb') as f:
        pickle.dump(recipe_features, f)

    print(f"{label} の特徴が成功裏に保存されました。")

@app.route('/ninnsiki/recipe_images', methods=['GET'])
def recipe_images():

    breadcrumbs = [
        {"name": "メインメニュー", "url": "/mainmenu/mainmenu"},
        {"name": "ごはん調教", "url": "/ninnsiki/recipe/images"}
    ]
    return render_template('ninnsiki/recipe_images.html', breadcrumbs=breadcrumbs)

@app.route('/ninnsiki/recipe_delete', methods=['GET'])
def recipe_delete():
    with open('recipe_features.pkl', 'rb') as f:
        recipe_features = pickle.load(f)

    # レシピのラベルを取得
    recipe_labels = sorted(recipe_features.keys())

    breadcrumbs = [
        {"name": "メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name": "ごはん調教" , "url":"/ninnsiki/recipe?images"},
        {"name": "レシピ消去", "url": "/ninnsiki/resipe_delete"}
    ]

    return render_template('ninnsiki/recipe_delete.html', recipe_labels=recipe_labels, breadcrumbs=breadcrumbs)

@app.route('/ninnsiki/delete_recipe', methods=['POST'])
def delete_recipe():
    label = request.form['label']  # 入力されたラベルを取得
    try:
        # 既存の料理特徴を読み込む
        with open('recipe_features.pkl', 'rb') as f:
            recipe_features = pickle.load(f)

        # 指定されたラベルを削除
        if label in recipe_features:
            del recipe_features[label]

            # 特徴をファイルに保存
            with open('recipe_features.pkl', 'wb') as f:
                pickle.dump(recipe_features, f)

            return render_template('ninnsiki/recipe_delete_success.html')
        else:
            return render_template('ninnsiki/recipe_delete_nothing.html')

    except Exception as e:
        return f"エラーが発生しました: {str(e)}"
    
@app.route('/ninnsiki/touroku_success', methods=['POST'])
def register_food():
    account_id = request.form['account_id']
    cuisine = request.form['cuisine']
    eat_date = datetime.now().date()

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute('INSERT INTO FOOD_DATA (ACCOUNT_ID, EAT_DATE, CUISINE) VALUES (?, ?, ?)',
                       (account_id, eat_date, cuisine))
        conn.commit()
    finally:
        conn.close()

    return redirect(url_for('touroku_success'))

@app.route('/ninnsiki/touroku_success')
def touroku_success():

    breadcrumbs = [
        {"name": "登録完了", "url": "/ninnsiki/touroku_success"}
    ]

    return render_template('ninnsiki/touroku_success.html', breadcrumbs=breadcrumbs)

@app.route('/ninnsiki/recipe_look', methods=['GET'])
def recipe_look():
    # 既存の料理特徴を読み込む
    with open('recipe_features.pkl', 'rb') as f:
        recipe_features = pickle.load(f)

    # レシピのラベルを取得
    recipe_labels = sorted(recipe_features.keys())

    breadcrumbs = [
        {"name":"メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name":"ごはん調教", "url":"/ninnsiki/recipe_images"},
        {"name":"レシピ一覧", "url":"/ninnsiki/recipe_look"}
    ]
    
    return render_template('ninnsiki/recipe_look.html', recipe_labels=recipe_labels, breadcrumbs=breadcrumbs)



@app.route('/ninnsyou/login', methods=['GET', 'POST'])
def login():
    mail_address = ""
    error_message = ""
    if request.method == 'POST':
        mail_address = request.form['mail_address']
        password = request.form['password']
        
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM ACCOUNT WHERE MAIL = ? AND PASS = ?", (mail_address, password))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user:
            session['account_id'] = user[0]
            session['account_name'] = user[1]
            if user[0] == 1:
                return redirect(url_for('master_check'))
            else:
                return redirect(url_for('mainmenu'))
        else:
            error_message = "ログインに失敗しました。アカウント名またはパスワードが間違っています。"

    return render_template('ninnsyou/login.html', mail_address=mail_address, error_message=error_message)

@app.route('/ninnsyou/master_check', methods=['GET', 'POST'])
def master_check():
    if request.method == 'POST':
        if request.form['confirm'] == 'yes':
            return redirect(url_for('mainmenu'))
        else:
            return redirect(url_for('login'))

    return render_template('ninnsyou/master_check.html')

@app.route('/ninnsyou/signup', methods=['GET'])
def sign_up():
    return render_template('ninnsyou/signup.html')

@app.route('/ninnsyou/signup', methods=['POST'])
def signup():
    account_name = request.form['account_name']
    mail_address = request.form['mail_address']
    password = request.form['password']

    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, mail_address):
        error_message = "メールアドレスの形式が正しくありません。"
        return render_template('ninnsyou/signup.html', account_name=account_name, mail_address=mail_address, error_message=error_message)

    # 入力値のバリデーション
    if len(account_name) > 10 or len(password) < 8 or len(password) > 20:
        error_message = "アカウント名は10桁以内、パスワードは8桁以上20桁以内で入力してね"
        return render_template('ninnsyou/signup.html', account_name=account_name, mail_address=mail_address, error_message=error_message)

    conn = get_db()
    cur = conn.cursor()
    # メールアドレスの重複チェック
    cur.execute("SELECT * FROM ACCOUNT WHERE MAIL = ?", (mail_address,))
    if cur.fetchone() is not None:
        cur.close()
        conn.close()
        error_message = "そのメールアドレスは既に使用されています。"
        return render_template('ninnsyou/signup.html', account_name=account_name, mail_address=mail_address, error_message=error_message)

    #新規登録
    cur.execute("INSERT INTO ACCOUNT (ACCOUNT_NAME, MAIL, PASS) VALUES (?, ?, ?)", (account_name, mail_address, password))
    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for('signup_success'))

@app.route('/ninnsyou/signup_success')
def signup_success():
    return render_template('ninnsyou/signup_success.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if request.method == 'POST':
        session.clear()
        return redirect(url_for('logout_success'))
    return render_template('ninnsyou/logout.html')

@app.route('/logout_success')
def logout_success():
    return render_template('ninnsyou/logout_success.html')

@app.context_processor
def inject_account_info():
    return {
        'account_name': session.get('account_name'),
        'account_id': session.get('account_id')
    }

@app.route('/mainmenu/mainmenu', methods=['GET', 'POST'])
def mainmenu():
    if 'account_name' in session:
        breadcrumbs = [
            {"name": "メインメニュー", "url": "/mainmenu/mainmenu"}
        ]
        return render_template('mainmenu/mainmenu.html', breadcrumbs=breadcrumbs)
    return redirect(url_for('login'))

@app.route('/master/account_look')
def account_look():
    if 'account_name' in session:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT ACCOUNT_ID, ACCOUNT_NAME, MAIL, PASS FROM ACCOUNT")
        accounts = cur.fetchall()
        cur.close()
        conn.close()

        breadcrumbs = [
            {"name": "メインメニュー", "url": "/mainmenu/mainmenu"},
            {"name": "アカウント閲覧", "url": "/master/account_look"}
        ]
        
        return render_template('master/account_look.html', accounts=accounts, breadcrumbs=breadcrumbs)

@app.route('/edit_account/<int:account_id>', methods=['POST'])
def edit_account(account_id):
    data = request.json
    account_name = data['account_name']
    mail_address = data['mail_address']
    password = data['password']

    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE ACCOUNT SET ACCOUNT_NAME = ?, MAIL = ?, PASS = ? WHERE ACCOUNT_ID = ?",
                (account_name, mail_address, password, account_id))
    conn.commit()
    return '', 204

@app.route('/master/account_delete/<int:account_id>', methods=['POST'])
def delete_account(account_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM ACCOUNT WHERE ACCOUNT_ID = ?", (account_id,))
    conn.commit()
    return '', 204

@app.route('/photo/photo_menu')
def photo_menu():

    breadcrumbs = [
        {"name": "メインメニュー", "url": "/mainmenu/mainmenu"},
        {"name": "フォトメニュー", "url": "/photo/photo_menu"}
    ]

    return render_template('photo/photo_menu.html',breadcrumbs=breadcrumbs)

@app.route('/photo/photo_take')
def photo_take():

    breadcrumbs = [
        {"name":"メインメニュー", "url": "/mainmenu/mainmenu"},
        {"name":"フォトメニュー", "url": "photo/photo_menu"},
        {"name":"撮影画面", "url": "/photo/photo_take"}
    ]
    return render_template('photo/photo_take.html',breadcrumbs=breadcrumbs)

def get_history():
    # データベースに接続
    conn = get_db()
    cursor = conn.cursor()

    # HISTORYテーブルからデータを取得
    cursor.execute("SELECT SUGG_ID, DAY, SUGG_txt FROM HISTORY ORDER BY SUGG_ID DESC")
    rows = cursor.fetchall()

    # 接続を閉じる
    conn.close()

    # データを返す（過去の提案）
    return rows

def save_to_history(sugg_txt):
    # データベースに接続
    conn = get_db()
    cursor = conn.cursor()

    # 提案内容をHISTORYテーブルに保存
    cursor.execute("INSERT INTO HISTORY (SUGG_txt) VALUES (?)", (sugg_txt,))

    # 変更を保存
    conn.commit()

    # 接続を閉じる
    conn.close()

@app.route('/sugg/sugg_menu')
def sugg_menu():

    breadcrumbs = [
        {"name":"メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name":"ごはん提案", "url":"/sugg/sugg_menu"}
    ]
    return render_template('sugg/sugg_menu.html',breadcrumbs=breadcrumbs)

@app.route('/generate', methods=['POST'])
def generate_content():
    # アカウントIDを取得
    account_id = session.get('account_id')

    # データベースからアレルギー情報を取得
    allergens = []
    try:
        # DB接続を取得
        connection = get_db()
        cursor = connection.cursor()
        cursor.execute("SELECT EGG, MILK, WHEAT, SHRIMP, CRAB, PEANUT, BUCKWHEAT FROM ALLERGEN WHERE ACCOUNT_ID = ?", (account_id,))
        row = cursor.fetchone()
        
        if row:
            # アレルギーがTrueの項目を抽出
            allergen_names = ["卵", "乳", "小麦", "えび", "かに", "ピーナッツ", "そば"]
            allergens = [allergen_names[i] for i, value in enumerate(row) if value]

    except Exception as e:
        return f"データベースエラー: {str(e)}"

    # アレルギー情報に基づいて定型文を生成
    if allergens:
        allergen_list = "、".join(allergens)
        content = f"今夜のごはんのおすすめを3つ提示してください。ただし、{allergen_list}を含まないものにしてください。"
    else:
        content = "今夜のごはんのおすすめを3つ提示してください。名前のみ"

    # Gemini APIに送信するリクエストデータ
    data = {
        "contents": [{
            "parts": [{
                "text": content
            }]
        }]
    }

    # APIリクエストを送信
    response = requests.post(GEMINI_API_URL, json=data, params={'key': API_KEY})

    # レスポンスの処理
    if response.status_code == 200:
        response_data = response.json()
        generated_content = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '生成に失敗しました。')
        
        save_to_history(generated_content)

        breadcrumbs = [
        {"name": "メインメニュー","url": "/mainmenu/mainmenu"},
        {"name": "ごはん提案", "url": "/sugg/sugg_menu"},
        {"name": "ごはん提案閲覧", "url":"/sugg/sugg_look"}
    ]
        return render_template('sugg/sugg_look.html', result=generated_content, breadcrumbs=breadcrumbs)
    else:
        error_message = f"エラーが発生しました: {response.status_code}"
        return render_template('sugg/sugg_look.html', result=error_message)
@app.route('/sugg/sugg_look')
def sugg_look():
    # このルートにアクセスしたとき、定型文を送信して結果を表示
    return generate_content()



@app.route('/sugg/sugg_hist') 
def sugg_hist():
    account_id = session.get('account_id')  # セッションからログイン中のアカウントIDを取得
    if not account_id:
        return redirect(url_for('home'))  # アカウントIDがない場合はホームにリダイレクト

    conn = get_db()  # データベース接続
    cursor = conn.cursor()

    # `HISTORY` テーブルから指定されたアカウントIDのデータを取得
    cursor.execute("""
    SELECT SUGG_ID, DAY, SUGG_txt
    FROM HISTORY
    WHERE ACCOUNT_ID = ?
    ORDER BY DAY DESC
    """, (account_id,))
    
    history = cursor.fetchall()  # 結果を取得
    conn.close()  # データベース接続を閉じる

    breadcrumbs = [
        {"name": "メインメニュー", "url": "/mainmenu/mainmenu"},
        {"name": "ごはん提案", "url": "/sugg/sugg_menu"},
        {"name": "過去のごはん提案", "url":"/sugg/sugg_hist"}
    ]

    # データをHTMLに渡して表示
    return render_template('sugg/sugg_hist.html', history=history, breadcrumbs=breadcrumbs)


@app.route('/sugg/eat_hist')
def eat_hist():
    account_id = session.get('account_id')  # セッションからログイン中のアカウントIDを取得
    if not account_id:
        return redirect(url_for('home'))  # アカウントIDがない場合はホームにリダイレクト

    conn = get_db()  # データベース接続
    cursor = conn.cursor()

    # `FOOD_DATA` テーブルから指定されたアカウントIDのデータを取得
    cursor.execute("""
    SELECT EAT_DATE, CUISINE
    FROM FOOD_DATA
    WHERE ACCOUNT_ID = ?
    ORDER BY EAT_DATE DESC
    """, (account_id,))
    
    food_data_list = cursor.fetchall()  # 結果を取得
    conn.close()  # データベース接続を閉じる

    breadcrumbs = [
        {"name": "メインメニュー", "url": "/mainmenu/mainmenu"},
        {"name": "ごはん提案", "url":"/sugg/sugg_menu"},
        {"name": "過去の食事記録", "url":"/sugg/eat?hist"}
    ]

    # データをHTMLに渡して表示
    return render_template('sugg/eat_hist.html', food_data_list=food_data_list, account_id=account_id, breadcrumbs=breadcrumbs)


# 本番ではこっちを検索して消せ
@app.route('/hiroba/area_gohan')
def area_gohan():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT P.POST_ID, A.ACCOUNT_ID, A.ACCOUNT_NAME, P.SENTENCE, P.PHOTO, P.CREATED_AT
    FROM POST P
    JOIN ACCOUNT A ON P.ACCOUNT_ID = A.ACCOUNT_ID
    ORDER BY P.POST_ID DESC
    """)
    
    posts = cursor.fetchall()
    
    conn.close()
    
    # フォーマットを変更して辞書リストを作成
    posts = [{'post_id': row[0], 'account_id': row[1], 'account_name': row[2], 'sentence': row[3], 'photo': row[4], 'created_at': row[5]} for row in posts]

    account_id = session.get('account_id')

    breadcrumbs = [
        {"name":"メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name":"ごはん広場","url":"/hiroba/area_gohan"}
    ]
    
    return render_template('hiroba/area_gohan.html', posts=posts, account_id=account_id,breadcrumbs=breadcrumbs)

@app.template_filter('add_hours')
def add_hours(value, hours):
    if value:
        if isinstance(value, str):
            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        return value + timedelta(hours=hours)
    return value

# @app.route('/hiroba/area_gohan')
# def area_gohan():
#     conn = get_db()
#     cursor = conn.cursor()

#     cursor.execute("""
#     SELECT P.POST_ID, A.ACCOUNT_ID, A.ACCOUNT_NAME, P.SENTENCE, P.PHOTO
#     FROM POST P
#     JOIN ACCOUNT A ON P.ACCOUNT_ID = A.ACCOUNT_ID
#     ORDER BY P.POST_ID DESC
#     """)

#     posts = cursor.fetchall()
#     conn.close()

#     # フォーマットを変更して辞書リストを作成
#     posts = [{'post_id': row[0], 'account_id': row[1], 'account_name': row[2], 'sentence': row[3], 'photo': row[4]} for row in posts]

#     account_id = session.get('account_id')

#     return render_template('hiroba/area_gohan.html', posts=posts, account_id=account_id)

@app.route('/hiroba/edit_post/<int:post_id>', methods=['POST'])
def edit_post(post_id):
    data = request.json
    sentence = data['sentence']  # 投稿文を受け取る
    # サニタイズを適用
    sentence = sanitize_post_content(sentence)
    # ここで flask.escape() を使ってエスケープ
    sentence = escape(sentence)  # HTML エスケープを行う
    # データベースに接続し、更新を行う
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE POST SET SENTENCE = ? WHERE POST_ID = ?", (sentence, post_id))  # サニタイズされた投稿文を保存
        conn.commit()
        app.logger.info(f"Post {post_id} updated successfully with sentence: {sentence}")
        return '', 204  # 成功時は204 No Contentを返す
    except Exception as e:
        logging.error(f"Error updating post: {e}")
        conn.rollback()
        return "Error updating post", 500  # エラー時には500を返す



#---------------------------サニタイズ処理関連------------------------------------------

import bleach
import re

# 許可するタグと属性
allowed_tags = ['b', 'i', 'u', 'em', 'strong', 'a', 'p', 'br']  # 許可するタグをリスト

def sanitize_post_content(content):
    """
    ユーザーが入力した投稿内容をサニタイズする関数
    - 不正なタグやスクリプトを除去
    - 許可されたタグだけを残す
    """
    # bleach.clean() で許可されたタグだけを残し、その他は削除
    sanitized_content = bleach.clean(content, tags=allowed_tags, strip=True)
    
    # style タグを削除
    sanitized_content = re.sub(r'<style.*?>.*?</style>', '', sanitized_content, flags=re.DOTALL)  # <style>タグ全体を削除
    
    # style 属性を削除（他の属性が残らないようにする）
    sanitized_content = re.sub(r' style="[^"]*"', '', sanitized_content)  # style 属性の削除
    
    # 中括弧内の内容（CSS など）を除去
    sanitized_content = re.sub(r'\{.*?\}', '', sanitized_content)  # CSS など、中括弧内の内容を除去

    return sanitized_content


#---------------------------------------------------------------------------------------

@app.route('/hiroba/post_gohan')
def post_gohan():

    breadcrumbs = [
        {"name":"メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name":"ごはん広場", "url":"/hiroba/area_gohan"},
        {"name":"投稿フォーム", "url":"/hiroba/post_gohan"}
    ]
    return render_template('hiroba/post_gohan.html', breadcrumbs=breadcrumbs)

# 英数字の桁数を変えたいときは k= の後の数値を変えてね
def generate_unique_filename(extension):
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return f"{random_str}.{extension}"


# 本番ではこっちを検索して消せ

@app.route('/hiroba/save_gohan_post', methods=['POST'])
def save_gohan_post():
    account_id = session.get('account_id')
    sentence = request.form['sentence']  # 投稿文を取得
    
    # サニタイズを適用し、flask.escapeでエスケープ
    sentence = escape(sentence)  # ここで HTML エスケープを行う
    
    photo = request.files['photo']
    
    if photo:
        # 元の拡張子のまま
        extension = photo.filename.rsplit('.', 1)[1].lower()
        unique_filename = generate_unique_filename(extension)
        
        # 保存
        photo_path = os.path.join(LOCAL_IMAGE_FOLDER, unique_filename)
        photo.save(photo_path)

        conn = get_db()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO POST (ACCOUNT_ID, SENTENCE, PHOTO) VALUES (?, ?, ?)",
                (account_id, sentence, unique_filename)  # エスケープされた投稿文を保存
            )
            conn.commit()
        finally:
            conn.close()
    
    return redirect(url_for('area_gohan'))


# @app.route('/hiroba/save_gohan_post', methods=['POST'])
# def save_gohan_post():
#     account_id = session.get('account_id')
#     sentence = request.form['sentence']
#     photo = request.files['photo']

#     if photo:
#         # 元の拡張子のまま
#         extension = photo.filename.rsplit('.', 1)[1].lower()
#         unique_filename = generate_unique_filename(extension)

#         # 保存先のパス
#         photo_path = os.path.join(LOCAL_IMAGE_FOLDER, unique_filename)
#         photo.save(photo_path)

#         # データベースに情報を保存
#         conn = get_db()
#         cursor = conn.cursor()

#         try:
#             cursor.execute(
#                 "INSERT INTO POST (ACCOUNT_ID, SENTENCE, PHOTO) VALUES (?, ?, ?)",
#                 (account_id, sentence, unique_filename)
#             )
#             conn.commit()
#         finally:
#             conn.close()

#     return redirect(url_for('area_gohan'))

import os

@app.route('/hiroba/delete_post/<int:post_id>', methods=['POST'])
def delete_post(post_id):
    conn = get_db()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT PHOTO FROM POST WHERE POST_ID = ?", (post_id,))
        photo_path = cur.fetchone()
        
        cur.execute("DELETE FROM POST WHERE POST_ID = ?", (post_id,))
        conn.commit()
        
        if photo_path and photo_path[0]:
            # ここは本番では検索して消せ(正確なパスに修正)
            # パスをフルで書く感じになるはず
            base_dir = os.path.dirname(__file__)
            full_path = os.path.join(base_dir, 'static', 'hiroba_img', photo_path[0])
            if os.path.exists(full_path):
                os.remove(full_path)
    
    finally:
        cur.close()
        conn.close()
    
    return redirect(url_for('area_gohan'))




# ーーーーーーーーーーアカウント設定ーーーーーーーーーー
@app.route('/acset/acct_set')
def acct_set():

    breadcrumbs = [
        {"name": "メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name": "アカウント設定", "url":"/acset/acct_set"}
    ]
    return render_template('acset/acct_set.html',breadcrumbs=breadcrumbs)

@app.route('/acset/allergy_new')
def allergy_new():

    breadcrumbs = [
        {"name":"メインメニュー","url":"/mainmenu/mainmenu"},
        {"name":"アカウント設定","url":"/acset/acct_set"},
        {"name":"アレルギー情報の更新", "url":"/acset/allergy_new"}
    ]
    
    return render_template('acset/allergy_new.html',breadcrumbs=breadcrumbs)

@app.route('/acset/new_allergy/<int:account_id>', methods=['GET', 'POST'])
def new_allergy(account_id):
    # チェックされたアレルギーの情報を取得
    egg = request.form.get('egg',False) == 'true'
    milk = request.form.get('milk',False) == 'true'
    wheat = request.form.get('wheat',False) == 'true'
    shrimp = request.form.get('shrimp',False) == 'true'
    crab = request.form.get('crab',False) == 'true'
    peanut = request.form.get('peanut',False) == 'true'
    buckwheat = request.form.get('buckwheat',False) == 'true'

    conn = get_db()
    # アレルギー情報を更新または挿入
    conn.execute('''
        INSERT OR REPLACE INTO ALLERGEN (account_id, egg, milk, wheat, shrimp, crab, peanut, buckwheat)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
        (account_id, egg, milk, wheat, shrimp, crab, peanut, buckwheat))
    
    sql = """
        SELECT
            CASE WHEN egg = 1 THEN '卵' ELSE NULL END AS allergy1,
            CASE WHEN milk = 1 THEN '牛乳' ELSE NULL END AS allergy2,
            CASE WHEN wheat = 1 THEN '小麦' ELSE NULL END AS allergy3,
            CASE WHEN shrimp = 1 THEN 'えび' ELSE NULL END AS allergy4,
            CASE WHEN crab = 1 THEN 'かに' ELSE NULL END AS allergy5,
            CASE WHEN peanut = 1 THEN '落花生' ELSE NULL END AS allergy6,
            CASE WHEN buckwheat = 1 THEN 'そば' ELSE NULL END AS allergy7
        FROM ALLERGEN
        WHERE account_id = ?;"""
    jiken = conn.execute(sql, (account_id,)).fetchone()
    conn.commit()
    conn.close()

    allergies = [allergy for allergy in jiken if allergy is not None]

    breadcrumbs = [
        {"name": "メインメニュー", "url":"/mainmenu/meinmenu"},
        {"name": "アカウント設定", "url":"/acset/acct_set"},
        {"name": "アレルギーの登録", "url":"/acset/allergy_set"}
    ]

    return render_template('acset/allergy_set.html',account_id=account_id,allergies=allergies,breadcrumbs=breadcrumbs)

@app.route('/acset/psd_change', methods=['GET','POST'])
def psd_change():

    breadcrumbs = [
        {"name":"メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name":"アカウント設定", "url":"/acset/acct_set"},
        {"name":"パスワード変更", "url":"/acset/psd_change"}
    ]
    return render_template('acset/psd_change.html', breadcrumbs=breadcrumbs)

@app.route('/change_psd/<int:account_id>', methods=['POST'])
def change_psd(account_id):
    error_message = ""
    password = request.form.get('password')
    password2 = request.form.get('passwordnew')
    password3 = request.form.get('passwordnew2')

    conn = get_db()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT * FROM ACCOUNT WHERE PASS = ?", (password,))
        acuser = cur.fetchone()

        if acuser:
            print("通過１")
        else:
            error_message = "入力されたパスワードが間違っています"
            return render_template('acset/psd_change.html', error_message=error_message)

        if password2 == password3:
            if len(password2) > 7 and len(password2) < 21:
                newpassword = password2
                cur.execute("UPDATE ACCOUNT SET PASS = ? WHERE ACCOUNT_ID = ?", (newpassword, account_id))
                conn.commit()
                return redirect(url_for('psd_changec'))
            else:
                error_message = "パスワードは8文字以上20文字以下で入力してください"
                return render_template('acset/psd_change.html', error_message=error_message)
        else:
            error_message = "新しく入力したパスワードのどちらかが間違っています"
            return render_template('acset/psd_change.html', error_message=error_message)
    except Exception as e:
        error_message = str(e)
        return render_template('acset/psd_change.html', error_message=error_message)
    finally:
        cur.close()
        conn.close()

@app.route('/acset/psd_changec')
def psd_changec():

    breadcrumbs = [
        {"name": "メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name": "アカウント設定", "url":"/acset/acct_set"},
        {"name": "パスワード変更", "url":"/acset/psd_change"},
        {"name": "パスワード変更完了", "url":"/acset_psd_changec"}
    ]
    return render_template('acset/psd_changec.html',breadcrumbs=breadcrumbs)

@app.route('/acset/acct_del', methods=['GET','POST'])
def acct_del():
    breadcrumbs = [
        {"name": "メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name": "アカウント設定", "url":"/acset/acct_set"},
        {"name": "アカウント削除", "url":"/acset/acct_del"}
    ]
    return render_template('acset/acct_del.html', breadcrumbs=breadcrumbs)

@app.route('/del_acct/<int:account_id>', methods=['POST'])
def del_acct(account_id):
    error_message = ""
    password = request.form.get('password')
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT PASS FROM ACCOUNT WHERE ACCOUNT_ID = ?", (account_id,))
        acuser = cur.fetchone()
        accuser = acuser[0]
        if accuser == password:
            return redirect(url_for('acct_del_con'))
        else:
            error_message = "入力されたパスワードが間違っています"

        breadcrumbs = [
        {"name": "メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name": "アカウント設定", "url":"/acset/acct_set"},
        {"name": "アカウント削除", "url":"/acset/acct_del"}
        ]
            
        return render_template('acset/acct_del.html', error_message=error_message,breadcrumbs=breadcrumbs)
    
    except Exception as e:
        error_message = str(e)

        breadcrumbs = [
        {"name": "メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name": "アカウント設定", "url":"/acset/acct_set"},
        {"name": "アカウント削除", "url":"/acset/acct_del"}
        ]

        return render_template('acset/acct_del.html', error_message=error_message, breadcrumbs=breadcrumbs)
    finally:
        cur.close()
        conn.close()
        
@app.route('/acset/acct_del_con')
def acct_del_con():
    return render_template('acset/acct_del_con.html')

@app.route('/acct_delete/<int:account_id>', methods=['POST'])
def acct_delete(account_id):
    
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("DELETE FROM ACCOUNT WHERE ACCOUNT_ID = ?", (account_id,))
    conn.commit()
    
    cur.close()
    conn.close()
    session.clear()
    return render_template('acset/acct_del_succ.html')

@app.route('/acset/acct_del_succ')
def acct_del_succ():
    return render_template('acset/acct_del_succ.html')

@app.route('/photo/photo_upload')
def photo_upload():

    breadcrumbs = [
        {"name": "メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name": "フォトメニュー", "url":"/photo/photo_menu"},
        {"name": "画像アップロード", "url":"/photo/photo_upload"}
    ]
    return render_template('photo/photo_upload.html',breadcrumbs=breadcrumbs)

@app.route('/photo/photo_recog')
def photo_recog():

    breadcrumbs = [
        {"name": "メインメニュー", "url":"/mainmenu/mainmenu"},
        {"name": "フォトメニュー", "url":"/photo/photo_menu"},
        {"name": "撮影画面", "url":"/photo/photo_take/"},
        {"name": "撮影結果画面", "url":"/photo/photo_recog"}
    ]
    return render_template('photo/photo_recog.html', breadcrumbs=breadcrumbs)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error/error.html'), 404

@app.errorhandler(405)
def internal_server_error(error):
    return render_template('error/error.html'), 405

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('error/error.html'), 500

if __name__ == '__main__':
    app.run(debug=True)