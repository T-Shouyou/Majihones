from flask import Flask, request, render_template
import boto3
import io
from PIL import Image, ImageDraw, ImageFont
import base64

app = Flask(__name__)
s3_client = boto3.client('s3')
rekognition_client = boto3.client('rekognition')

# 画像にバウンディングボックスを描画する関数
def display_image(bucket, photo, response):
    # S3バケットから画像をロード
    s3_response = s3_client.get_object(Bucket=bucket, Key=photo.filename)
    stream = io.BytesIO(s3_response['Body'].read())
    image = Image.open(stream)

    imgWidth, imgHeight = image.size
    draw = ImageDraw.Draw(image)

    # カスタムラベルを検出して描画
    for customLabel in response['CustomLabels']:
        if 'Geometry' in customLabel:
            box = customLabel['Geometry']['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']

            draw.text((left, top), customLabel['Name'], fill='#00d400', font=ImageFont.load_default())
            points = [
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top)
            ]
            draw.line(points, fill='#00d400', width=5)

    # 修正した画像をBase64に変換して返す
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

# RekognitionのCustom labelsを判定する関数
def detect_custom_labels(model, bucket, photo):
    return rekognition_client.detect_custom_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': photo.filename}},
        MinConfidence=95,
        ProjectVersionArn=model
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            try:
                bucket = 'custom-labels-console-us-east-1-3b809520ae'
                s3_client.upload_fileobj(file, bucket, file.filename)

                model = 'arn:aws:rekognition:us-east-1:552409373703:project/SakanaFriends/version/SakanaFriends.2024-09-27T12.23.47/1727407428069'
                response = detect_custom_labels(model, bucket, file)

                # 判定された食材名を取得
                food_names = ', '.join(label['Name'] for label in response['CustomLabels'])

                # バウンディングボックスを描画した画像を取得
                modified_image = display_image(bucket, file, response)

                # 結果ページをレンダリング
                return render_template('success.html', modified_image=modified_image, food_names=food_names)
            except Exception as e:
                error = f'Error occurred: {e}'
        else:
            error = 'No file selected'

    return render_template('upload.html', error=error)

if __name__ == "__main__":
    app.run(debug=True)
