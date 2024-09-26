import cv2
import numpy as np
import pickle
import os

# 各料理の特徴を格納する辞書
recipe_features = {}

# 料理画像のパスとその名前
recipe_images = {
    'ramen': 'recipe_images/ramen.jpg',
    'sushi': 'recipe_images/sushi.jpg',
    'curry': 'recipe_images/curry.jpg'
}

# 特徴を計算して辞書に保存
for dish, img_path in recipe_images.items():
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    histogram = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    recipe_features[dish] = histogram

# 特徴をファイルに保存
with open('recipe_features.pkl', 'wb') as f:
    pickle.dump(recipe_features, f)

print("特徴が成功裏に保存されました。")
