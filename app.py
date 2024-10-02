from flask import Flask, request, render_template, jsonify
import boto3
import io
from PIL import Image, ImageDraw, ImageFont
import base64
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

app = Flask(__name__)

# 環境変数からAWSの認証情報を取得
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN')  # セッショントークンがある場合
)

rekognition_client = boto3.client(
    'rekognition',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN')  # セッショントークンがある場合
)

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

# プリサインドURLを生成する関数
def generate_presigned_url(bucket_name, object_name, expiration=3600):
    try:
        response = s3_client.generate_presigned_url('put_object',
                                                    Params={'Bucket': bucket_name, 'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
    return response

# プリサインドURLのAPIエンドポイント
@app.route('/generate_presigned_url', methods=['POST'])
def generate_presigned_url_route():
    data = request.get_json()
    file_name = data['filename']
    bucket_name = 'custom-labels-console-us-east-1-0df316a052'

    presigned_url = generate_presigned_url(bucket_name, file_name)

    if presigned_url:
        return jsonify({'url': presigned_url})
    else:
        return jsonify({'error': 'Could not generate URL'}), 500

# メインの画像アップロードページ
@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file is not None and file.filename != '':
            try:
                bucket = 'custom-labels-console-us-east-1-0df316a052'
                s3_client.upload_fileobj(file, bucket, file.filename)

                model = 'arn:aws:rekognition:us-east-1:898322155510:project/MAJIHONE/version/MAJIHONE.2024-09-30T13.53.23/1727672002204'
                response = detect_custom_labels(model, bucket, file)

                # 判定された食材名を取得
                food_names = ', '.join(label['Name'] for label in response['CustomLabels'])

                # バウンディングボックスを描画した画像を取得
                modified_image = display_image(bucket, file, response)

                # 結果ページをレンダリング
                return render_template('success.html', modified_image=modified_image, food_names=food_names)
            except Exception as e:
                error = f'Failed to process the uploaded image: {e}'
            return render_template('error.html', error=error)  # エラー時にエラーページを表示

    return render_template('upload.html', error=error)

if __name__ == "__main__":
    app.run(debug=True)
