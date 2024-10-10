import cv2
import numpy as np
import pickle
import boto3
import os

# 各料理の特徴を格納する辞書
recipe_features = {}

# S3の設定
s3_client = boto3.client('s3', region_name='us-east-1')  # リージョンを指定
bucket_name = 'gazou'  # バケット名

# S3から料理画像のパスを取得
def get_recipe_images_from_s3():
    recipe_images = {}
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='recipe_images/')
    
    if 'Contents' in response:
        for obj in response['Contents']:
            # ファイル名を取得
            file_name = os.path.basename(obj['Key'])
            # ラベル（ファイル名から拡張子を除いた部分）を取得
            label = os.path.splitext(file_name)[0]
            recipe_images[label] = obj['Key']  # ラベルとパスを辞書に追加
    return recipe_images

# 特徴を計算して辞書に保存
recipe_images = get_recipe_images_from_s3()  # S3から画像を取得

for label, img_path in recipe_images.items():
    # S3から画像をダウンロード
    img = s3_client.get_object(Bucket=bucket_name, Key=img_path)['Body'].read()
    img_array = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is not None:  # 画像が正しく読み込まれたか確認
        img = cv2.resize(img, (150, 150))
        histogram = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram)
        if label in recipe_features:
            recipe_features[label].append(histogram)  # 既存のリストに追加
        else:
            recipe_features[label] = [histogram]  # 新規ラベルの場合リストを作成
    else:
        print(f"Error: Could not read image {img_path}")  # エラーメッセージ

# 特徴をファイルに保存
with open('recipe_features.pkl', 'wb') as f:
    pickle.dump(recipe_features, f)

print("特徴が成功裏に保存されました。")
