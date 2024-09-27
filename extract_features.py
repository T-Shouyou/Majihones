import cv2
import numpy as np
import pickle
import os

# 各料理の特徴を格納する辞書
recipe_features = {}

# 料理画像のパスとその名前
recipe_images = {
    'ramen': ['recipe_images/ramen.jpg', 'recipe_images/20240221_133857.jpg'],
    'sushi': ['recipe_images/sushi.jpg'],
    'curry': ['recipe_images/curry.jpg'],
    'bread': ['recipe_images/20230311_114615.jpg'],
    'donuts': ['recipe_images/20230718_181532.jpg', 'recipe_images/20240403_154901.jpg', 'recipe_images/20240405_172517.jpg'],
    'drink': ['recipe_images/20230311_103429.jpg', 'recipe_images/20230311_123817.jpg', 'recipe_images/20240416_154330.jpg', 'recipe_images/IMG_20230711_160123340_HDR.jpg'],
    'icecream': ['recipe_images/20230519_155045.jpg'],
    'kokomiice': ['recipe_images/20240221_134834.jpg'],
    'misokin': ['recipe_images/20240526_215040.jpg', 'recipe_images/20240825_182132.jpg'],
    'sweets': ['recipe_images/20230925_125003.jpg', 'recipe_images/IMG_20221126_230134828.jpg', 'recipe_images/IMG_20230918_214334897.jpg', 'recipe_images/IMG_20230924_164716410.jpg', 'recipe_images/IMG_20240701_131312884.jpg', 'recipe_images/puding.jpg'],
    'teisyoku': ['recipe_images/20230405_135330.jpg', 'recipe_images/20230925_122122.jpg', 'recipe_images/20240403_124335.jpg', 'recipe_images/IMG_20230705_132136976.jpg', 'recipe_images/IMG_20230925_122033643.jpg', 'recipe_images/IMG_20230930_191047340.jpg', 'recipe_images/IMG_20231004_131559850.jpg'],
}

# 特徴を計算して辞書に保存
for dish, img_paths in recipe_images.items():  # img_pathsに修正
    histograms = []  # 各料理のヒストグラムを格納するリスト
    for img_path in img_paths:  # 各画像をループ
        img = cv2.imread(img_path)
        if img is not None:  # 画像が正しく読み込まれたか確認
            img = cv2.resize(img, (150, 150))
            histogram = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(histogram, histogram)
            histograms.append(histogram)  # ヒストグラムをリストに追加
        else:
            print(f"Error: Could not read image {img_path}")  # エラーメッセージ
    recipe_features[dish] = np.mean(histograms, axis=0)  # 各料理のヒストグラムの平均を保存

# 特徴をファイルに保存
with open('recipe_features.pkl', 'wb') as f:
    pickle.dump(recipe_features, f)

print("特徴が成功裏に保存されました。")
