import cv2
import numpy as np
import pickle
import boto3  # S3用のライブラリ
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_mysqldb import MySQL
import os
import secrets

app = Flask(__name__)

# シークレットキーの設定
app.secret_key = secrets.token_hex(16)  # セキュリティのためのシークレットキー

# MySQLの設定
app.config['MYSQL_HOST'] = 'UminekoSakana.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'UminekoSakana'
app.config['MYSQL_PASSWORD'] = 'KounoFriends'
app.config['MYSQL_DB'] = 'UminekoSakana$default'

# MySQLの初期化
mysql = MySQL(app)

# S3クライアントの初期化
s3_client = boto3.client('s3', region_name='us-east-1')  # リージョンを指定

# 事前に計算した料理の特徴をロード
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
    return render_template('ninnsiki/recipe_images.html')

@app.route('/ninnsiki/recipe_delete', methods=['GET'])
def recipe_delete():
    return render_template('ninnsiki/recipe_delete.html')

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

            return render_template('ninnski/recipe_delete_success.html')
        else:
            return "指定された料理名は存在しません。"

    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

@app.route('/ninnsyou/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        account_name = request.form['account_name']
        password = request.form['password']
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM ACCOUNT WHERE ACCOUNT_NAME = %s AND PASS = %s", (account_name, password))
        user = cur.fetchone()
        cur.close()

        if user:
            session['account_name'] = account_name  # セッションにアカウント名を保存
            return redirect(url_for('main_menu'))  # ログイン成功時にメインメニューへリダイレクト
        else:
            return "ログインに失敗しました。アカウント名またはパスワードが間違っています。"

    return render_template('ninnsyou/login.html')


@app.route('/ninnsyou/signup')
def sign_up():
    return render_template('ninnsyou/signup.html')  # 新規登録ページを表示

@app.route('/ninnsyou/signup', methods=['POST','GET'])
def signup():
    account_name = request.form['account_name']
    mail_address = request.form['mail_address']
    password = request.form['password']

    # 入力値のバリデーション
    if len(account_name) > 10 or len(mail_address) > 25 or len(password) < 8 or len(password) > 20:
        return "入力値が不正です。アカウント名は10桁以内、メールアドレスは25桁以内、パスワードは8桁以上20桁以内で入力してください。"

    cur = mysql.connection.cursor()
    # アカウント名の重複チェック
    cur.execute("SELECT * FROM ACCOUNT WHERE ACCOUNT_NAME = %s", (account_name,))
    if cur.fetchone() is not None:
        cur.close()
        return "そのアカウント名は既に使用されています。"

    # データベースに新規登録
    cur.execute("INSERT INTO ACCOUNT (ACCOUNT_NAME, MAIL, PASS) VALUES (%s, %s, %s)", (account_name, mail_address, password))
    mysql.connection.commit()
    cur.close()

    return redirect(url_for('login'))  # 登録後にログインページへリダイレクト

@app.route('/ninnsyou/signup_success')
def signup_success():
    return render_template('ninnsyou/signup_success.html')

@app.route('/ninnsyou/logout')
def logout():
    session.pop('account_name', None)  # セッションからアカウント名を削除
    return redirect(url_for('login'))  # ログアウト後にログイン画面へリダイレクト

@app.route('/mainmenu/mainmenu')
def main_menu():
    return render_template('mainmenu/mainmenu.html', account_name=session.get('account_name'))

if __name__ == '__main__':
    app.run(debug=True)
