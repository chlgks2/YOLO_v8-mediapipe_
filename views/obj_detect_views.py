from flask import Blueprint, Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
import cv2
from ultralytics import YOLO
from gtts import gTTS
import numpy as np
import os
import base64
import sqlite3
import time
from os import listdir, path
from os.path import isfile, join
# from flask import current_app
from gugu_beta_master import create_app , mail

import datetime  
from flask_mail import Message

from collections import Counter

bp = Blueprint('main', __name__)

DATABASE = 'labels.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS labels (id INTEGER PRIMARY KEY, label TEXT, inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, time_information TEXT)''')
        conn.commit()


init_db()

# app = Flask(__name__)
# CORS(app)

# OpenAI API 키 설정
openai.api_key = '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'
model = YOLO('best.pt')
AUDIO_DIR = 'audio_files'

if not os.path.exists(AUDIO_DIR):
    os.mkdir(AUDIO_DIR)

def get_gpt_response(label):
    prompt = f"'{label}'이라는 단어를 한국어로 번역한 후에 어떤 물건인지 5살 아이가 이해할 수 있도록 2줄이내 문장으로 반말로 설명해줘"
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=200)
    translated_and_explained = response.choices[0].text.strip()
    return translated_and_explained

def generate_audio_from_text(text, lang='ko'):
    tts = gTTS(text=text, lang=lang, slow=False)
    # 현재 시간을 이용해 고유한 파일명을 생성합니다.
    filename = f"audio_{int(time.time())}.mp3"
    path = os.path.join(AUDIO_DIR, filename)
    tts.save(path)
    return filename


@bp.route('/detect', methods=['POST'])
def detect():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image_np = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    results = model.predict(source=image, save=True, conf=0.25)
    result = results[0]

    # 설명을 생성
    detected_boxes = result.boxes
    non_person_conf_values = [conf for i, conf in enumerate(detected_boxes.conf) if result.names[int(detected_boxes.cls[i])] != 'person']
    non_person_classes = [cls for cls in detected_boxes.cls if result.names[int(cls)] != 'person']

    if non_person_conf_values:
        highest_conf_idx = non_person_conf_values.index(max(non_person_conf_values))
        highest_conf_label = result.names[int(non_person_classes[highest_conf_idx])]
        explanation = f"이것은 '{highest_conf_label}'야. " + get_gpt_response(highest_conf_label)
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 현재 시간을 문자열로 변환

        save_to_db(highest_conf_label, current_time)
    else:
        explanation = "아무것도 탐지되지 않았습니다."

    # 가장 최근에 생성된 폴더의 경로를 가져오는 코드
    base_folder = "C:/Users/user/lecture/Craw_A/torch/2023_08_pj/yolo_tts/mom/gugu_beta-master (1)/gugu_beta_master/runs/detect"
                   
    all_subfolders = [f for f in listdir(base_folder) if path.isdir(path.join(base_folder, f)) and "predict" in f]
    all_subfolders.sort()  # 폴더 이름을 기준으로 정렬
    latest_subfolder = all_subfolders[-1]  # 가장 마지막 폴더 선택
    predicted_image_folder = path.join(base_folder, latest_subfolder)
    
    onlyfiles = [f for f in listdir(predicted_image_folder) if path.isfile(path.join(predicted_image_folder, f))]
    predicted_image_path = path.join(predicted_image_folder, onlyfiles[-1])  # 마지막에 생성된 파일을 선택

    # 이미지를 읽고 Base64로 인코딩
    detected_image = cv2.imread(predicted_image_path)
    retval, buffer = cv2.imencode('.jpg', detected_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    audio_path = generate_audio_from_text(explanation)
    audio_filename = audio_path.split("/")[-1]

    return jsonify({"explanation": explanation, "image": image_base64, "audio_path": audio_filename})



def save_to_db(label, time_information):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO labels (label, time_information) VALUES (?, ?)", (label, time_information))
        conn.commit()

@bp.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


@bp.route('/')
def intro():
    return render_template('intro.html')


@bp.route('/obdetect')
def object_detect():
    return render_template('object_detect.html')


@bp.route('/result_chart')
def result_chart():
    return render_template('result_chart.html')

def read_html_template(template_name):
    template_path = os.path.join(bp.root_path, 'templates', template_name)
    with open(template_path, 'r', encoding='utf-8') as template_file:
        return template_file.read()
    
# 날짜별 레이블 빈도수 데이터를 가져오는 함수
def get_label_counts_by_date(date):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT label, COUNT(*) as count FROM labels WHERE time_information LIKE ? GROUP BY label", (f"{date}%",))
        label_counts = [{"label": row[0], "count": row[1]} for row in cursor.fetchall()]
        print(label_counts)
        return label_counts
    
@bp.route('/chart_data/<date>')
def chart_data(date):
    label_counts = get_label_counts_by_date(date)
    return jsonify(label_counts)

# if __name__ == '__main__':
#     app.run()

# mail = Mail()

# @bp.route('/result', methods=['GET', 'POST'])
# def result():
#     if request.method == 'POST':
#         email = request.form.get('email')
        

#         with sqlite3.connect(DATABASE) as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT label FROM labels")
#             all_labels = [row[0] for row in cursor.fetchall()]


#         label_counts = Counter(all_labels)
#         most_common_labels = label_counts.most_common(5)


#         message = Message(subject="우리애의 관심물건 탑 5 ", recipients=[email], sender="chlgks22@gmail.com")
#         message.body = "\n".join([f"{label} - {count} occurrences" for label, count in most_common_labels])
        
#         message.body += "\n \n 더많은 정보가 알고싶으면 눌러보던가 :  http://127.0.0.1:5000/result_chart !"
#         mail.send(message)

#     return render_template('result copy.html')
result_blueprint = Blueprint('result', __name__)

@result_blueprint.route('/result', methods=['GET', 'POST'])
def result():
    most_common_labels = []  # 먼저 이 변수를 선언해야 합니다.

    if request.method == 'POST':
        email = request.form.get('email')  # 사용자가 입력한 이메일 주소를 가져옵니다.

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT label FROM labels")
            all_labels = [row[0] for row in cursor.fetchall()]

        label_counts = Counter(all_labels)
        most_common_labels = label_counts.most_common(5)

        # 이메일 메시지를 구성합니다.
        message = Message(subject="우리애의 관심물건 탑 5 ", recipients=[email], sender="chlgks22@gmail.com")
        message.body = "\n".join([f"{label} - {count} occurrences" for label, count in most_common_labels])
        message.body += "\n \n 더 많은 정보가 알고 싶으면 눌러보던가 :  http://127.0.0.1:5000/result_chart !"

        # 이메일을 송신합니다.
        mail.send(message)

    return render_template('result copy.html', most_common_labels=most_common_labels)  # 이렇게 변수를 템플릿에 전달합니다

# 안될때 해결법
# 환경 변수 설정: smtplib가 내부적으로 사용하는 환경 변수를 변경하여 문제를 해결할 수 있습니다. 터미널이나 명령 프롬프트를 열고 다음 명령을 실행해 보세요:

# bash
# Copy code
# set PYTHONUNBUFFERED=1

#디버그끄셈