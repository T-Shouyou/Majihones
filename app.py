import cv2
import numpy as np
import pickle
import boto3  # S3用のライブラリ
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import sqlite3  # SQLite用のライブラリ
import os
import secrets

app = Flask(__name__)

# シークレットキーの設定
app.secret_key = secrets.token_hex(16)  # セキュリティのためのシークレットキー

# SQLiteデータベースの設定
DATABASE = 'mydatabase.db'  # SQLiteデータベースのファイル名

def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn

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
            session['account_name'] = user[1]  # セッションにアカウント名を保存
            return redirect(url_for('mainmenu'))  # ログイン成功時にメインメニューへリダイレクト
        else:
            return "ログインに失敗しました。アカウント名またはパスワードが間違っています。"

    return render_template('ninnsyou/login.html')

@app.route('/ninnsyou/signup', methods=['GET'])
def sign_up():
    return render_template('ninnsyou/signup.html')  # 新規登録ページを表示

@app.route('/ninnsyou/signup', methods=['POST'])
def signup():
    account_name = request.form['account_name']
    mail_address = request.form['mail_address']
    password = request.form['password']

    # 入力値のバリデーション
    if len(account_name) > 10 or len(password) < 8 or len(password) > 20:
        return "アカウント名は10桁以内、パスワードは8桁以上20桁以内で入力してね"

    conn = get_db()
    cur = conn.cursor()
    # メールアドレスの重複チェック
    cur.execute("SELECT * FROM ACCOUNT WHERE MAIL = ?", (mail_address,))
    if cur.fetchone() is not None:
        cur.close()
        conn.close()
        return "そのメールアドレスは既に使用されています。"

    # データベースに新規登録
    cur.execute("INSERT INTO ACCOUNT (ACCOUNT_NAME, MAIL, PASS) VALUES (?, ?, ?)", (account_name, mail_address, password))
    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for('login'))  # 登録後にログインページへリダイレクト

@app.route('/ninnsyou/signup_success')
def signup_success():
    return render_template('ninnsyou/signup_success.html')

@app.route('/ninnsyou/logout')
def logout():
    session.pop('account_name', None)  # セッションからアカウント名を削除
    return redirect(url_for('login'))  # ログアウト後にログイン画面へリダイレクト

@app.route('/mainmanu/mainmenu')
def mainmenu():
    if 'account_name' in session:
        account_id = session.get('account_id')
        return render_template('mainmenu/mainmenu.html', account_name=session['account_name'], account_id=account_id)  # ログイン中のアカウント名を表示
    return redirect(url_for('login'))  # 未ログインの場合はログインページへリダイレクト

@app.route('/master/account_look')
def account_look():
    # ユーザーがログインしているかを確認
    if 'account_name' in session:
        # 以下の条件をコメントアウトして残しておく
        # if session.get('account_id') == 1:  # ACCOUNT_IDが1かどうかを確認
            conn = get_db()
            cur = conn.cursor()
            # すべてのアカウント情報を取得
            cur.execute("SELECT ACCOUNT_ID, ACCOUNT_NAME, MAIL, PASS FROM ACCOUNT")
            accounts = cur.fetchall()  # アカウント情報のリストを取得
            cur.close()
            conn.close()
            
            return render_template('master/account_look.html', accounts=accounts)  # アカウント情報をテンプレートに渡す

    return redirect(url_for('login'))  # 未ログインの場合はログインページへリダイレクト


@app.route('/photo/photo_menu')
def photo_menu():
    return render_template('photo/photo_menu.html', account_name=session['account_name'])  # ごはん撮影メニューを表示

@app.route('/photo/photo_take')
def photo_take():
    return render_template('photo/photo_take.html', account_name=session['account_name'])  # ごはん撮影メニューを表示

if __name__ == '__main__':
    app.run(debug=True)