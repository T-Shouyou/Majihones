from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# モデルの構築
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 二値分類の場合

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 画像データの準備
train_data_dir = 'C:/Users/h_kono/Desktop/sakana/Majihones/static/train'  # 修正
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

# モデルの訓練
model.fit(train_generator, epochs=10)
