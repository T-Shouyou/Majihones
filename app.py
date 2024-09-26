from flask import Flask, request, render_template
from datetime import datetime
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

# モデルの読み込み
model = load_model('model.h5')

# 食材のクラスラベル（例）
labels = ["ごはん", "野菜", "肉", "魚", "果物"]

# 毎回エラー表示は不安なので訪れた時刻でも表示しておく
@app.route("/")
def saisyo():
    now = datetime.now().replace(microsecond=0)
    return f"""
        <h2>このページに訪れた時刻は {now}<br>上のURLバーを正しく書き換えて。</h2>
        <a href='/gohann'>gohann に遷移</a>
    """

@app.route("/gohann", methods=['GET', 'POST'])
def gohann():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image provided", 400
        
        file = request.files['image']
        
        # 画像を読み込む
        img = Image.open(file.stream)
        img = img.resize((224, 224))  # モデルに合わせてサイズを変更
        img_array = img_to_array(img) / 255.0  # 正規化
        img_array = np.expand_dims(img_array, axis=0)  # バッチ次元を追加

        # 予測
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        label = labels[class_index]

        return f"認識された食材: {label}"

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
