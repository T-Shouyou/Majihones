import cv2
import numpy as np
import pickle  # これを追加
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# 事前に計算した料理の特徴をロード
with open('recipe_features.pkl', 'rb') as f:
    recipe_features = pickle.load(f)

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    histogram = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram

def identify_dish(image_path):
    features = process_image(image_path)
    min_distance = float('inf')
    predicted_dish = '不明な料理名'

    # 既知の料理特徴と比較
    for dish, recipe_hist in recipe_features.items():
        distance = np.linalg.norm(features - recipe_hist)
        if distance < min_distance:
            min_distance = distance
            predicted_dish = dish

    return f'予測された料理名: {predicted_dish}'

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_path = f"uploads/{file.filename}"
    file.save(image_path)
    
    predicted_label = identify_dish(image_path)
    return jsonify(predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
