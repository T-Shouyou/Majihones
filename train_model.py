import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 画像データの前処理を設定
train_datagen = ImageDataGenerator(rescale=1./255)

# ディレクトリのパスを設定（ここにGitHubのリポジトリ内のパスを指定）
train_directory = 'static/train'  # 実際の画像があるパスに変更

# データを生成する
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(224, 224),  # 画像のサイズ
    batch_size=32,  # バッチサイズ
    class_mode='categorical'  # 分類のモード
)

# モデルの構築
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # クラス数はデータセットに応じて調整
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデルの訓練
model.fit(train_generator, epochs=10)  # エポック数は必要に応じて調整

# モデルを保存
model.save('model.h5')
