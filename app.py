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
from datetime import datetime 


app = Flask(__name__)

app.secret_key = secrets.token_hex(16)

# 自分のパソコンで実行する際の画像の保存先のパス、本番では検索して消せ
LOCAL_IMAGE_FOLDER = 'static/hiroba_img'

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
    return render_template('ninnsiki/touroku_success.html')

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
            return redirect(url_for('mainmenu'))
        else:
            error_message = "ログインに失敗しました。アカウント名またはパスワードが間違っています。"

    return render_template('ninnsyou/login.html', mail_address=mail_address, error_message=error_message)

@app.route('/ninnsyou/signup', methods=['GET'])
def sign_up():
    return render_template('ninnsyou/signup.html')

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

    #新規登録
    cur.execute("INSERT INTO ACCOUNT (ACCOUNT_NAME, MAIL, PASS) VALUES (?, ?, ?)", (account_name, mail_address, password))
    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for('login'))

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

@app.route('/mainmenu/mainmenu')
def mainmenu():
    if 'account_name' in session:
        return render_template('mainmenu/mainmenu.html')
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
        
        return render_template('master/account_look.html', accounts=accounts)

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
    return render_template('photo/photo_menu.html')

@app.route('/photo/photo_take')
def photo_take():
    return render_template('photo/photo_take.html')

@app.route('/sugg/sugg_menu')
def sugg_menu():
    return render_template('sugg/sugg_menu.html')

@app.route('/sugg/sugg_look')
def sugg_look():
    return render_template('sugg/sugg_look.html')

@app.route('/sugg/sugg_hist')
def sugg_hist():
    return render_template('sugg/sugg_hist.html')

@app.route('/hiroba/area_gohan')
def area_gohan():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT P.POST_ID, A.ACCOUNT_ID, A.ACCOUNT_NAME, P.SENTENCE, P.PHOTO 
    FROM POST P
    JOIN ACCOUNT A ON P.ACCOUNT_ID = A.ACCOUNT_ID
    ORDER BY P.POST_ID DESC
    """)
    
    posts = cursor.fetchall()
    
    conn.close()
    
    # フォーマットを変更して辞書リストを作成
    posts = [{'post_id': row[0], 'account_id': row[1], 'account_name': row[2], 'sentence': row[3], 'photo': row[4]} for row in posts]

    account_id = session.get('account_id')
    
    return render_template('hiroba/area_gohan.html', posts=posts, account_id=account_id)

@app.route('/hiroba/edit_post/<int:post_id>', methods=['POST'])
def edit_post(post_id):
    data = request.json
    sentence = data['sentence']

    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE POST SET SENTENCE = ? WHERE POST_ID = ?", (sentence, post_id))
    conn.commit()
    return '', 204

@app.route('/hiroba/post_gohan')
def post_gohan():
    return render_template('hiroba/post_gohan.html')

# 英数字の桁数を変えたいときは k= の後の数値を変えてね
def generate_unique_filename(extension):
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return f"{random_str}.{extension}"

@app.route('/hiroba/save_gohan_post', methods=['POST'])
def save_gohan_post():
    account_id = session.get('account_id')
    sentence = request.form['sentence']
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
                (account_id, sentence, unique_filename)
            )
            conn.commit()
        finally:
            conn.close()
    
    return redirect(url_for('area_gohan'))


# ーーーーーーーーーーアカウント設定ーーーーーーーーーー
@app.route('/acset/acct_set')
def acct_set():
    return render_template('acset/acct_set.html')

@app.route('/acset/allergy_new')
def allergy_new():
    
    return render_template('acset/allergy_new.html')

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
    
    print(egg,milk,wheat,shrimp,crab,peanut,buckwheat)

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

    print(allergies)
    return render_template('acset/allergy_set.html',account_id=account_id,allergies=allergies)

@app.route('/acset/psd_change', methods=['GET','POST'])
def psd_change():
    return render_template('acset/psd_change.html')

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
    return render_template('acset/psd_changec.html')

@app.route('/acset/acct_del')
def acct_del():
    return render_template('acset/acct_del.html')

# -------------------------------------------------------

@app.route('/photo/photo_upload')
def photo_upload():
    return render_template('photo/photo_upload.html')

@app.route('/photo/photo_recog')
def photo_recog():
    return render_template('photo/photo_recog.html')

if __name__ == '__main__':
    app.run(debug=True)
