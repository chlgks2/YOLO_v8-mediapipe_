import random
import time
from flask import Blueprint, Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import pygame

bp = Blueprint('quiz', __name__, url_prefix='/')

max_num_hands = 1
gesture = { 0:'nothing' , 1:'point', 2:'v', 3:'three', 4:'four', 5:'five' }

df = pd.read_csv('finger_recognition.csv', header=None)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x = x.to_numpy().astype(np.float32)
y = y.to_numpy().astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(x, cv2.ml.ROW_SAMPLE, y)


def generate_frames():
    mp_hands = mp.solutions.hands

    # 한글 단어 표시
    def myPutText(src, text, pos, font_size, font_color):
        img_pil = Image.fromarray(src)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype('fonts/H2GTRM.TTF', font_size)
        lines = text.split('\n')  # 줄바꿈 문자('\n')을 기준 텍스트 나눔

        y_offset = pos[1]
        for line in lines:
            draw.text((pos[0], y_offset), line, font=font, fill=font_color)
            y_offset += font_size + 5  # 줄 간격 조절
        return np.array(img_pil)

    FONT_SIZE = 32
    FONT_COLOR = (0, 0, 0)
    RECTANGLE_COLOR = (255, 255, 255)
    THICKNESS = 4

    LOOPY = (200, 182, 255)

    # 퀴즈용 단어
    kids_words = ['신발', '포도', '노트북', '컵', '숟가락', '호랑이', '연필', '마우스', '휴지', '청소기', '딸기', 
                  '빗', '후라이팬', '믹서기', '개', '시계', '창문', '전자레인지', '핸드폰', '파인애플', '소파',
                  '모자', '포크', '키위', '의자', '이어폰', '리모콘', '비누', '냉장고', '필통'] 
    kids_words_eng = ['shoes', 'grape', 'laptop', 'cup', 'spoon', 'tiger', 'pencil', 'mouse', 'tissue', 'vacuum_cleaner', 'strawberry', 
                      'comb', 'frying_pan', 'blender', 'dog', 'clock', 'window', 'microwave', 'cellphone', 'pineapple', 'sofa',
                      'hat', 'fork', 'kiwi', 'chair', 'earphones', 'remote_control', 'soap', 'refrigerator', 'pencil_case']
    game_words_number = 5
    game_random_words = random.sample(kids_words, game_words_number)
    test_word = random.choice(game_random_words)

    # test 단어의 이미지 불러오기
    def get_img():
        test_word_idx = kids_words.index(test_word)
        test_word_eng = kids_words_eng[test_word_idx]

        # 이미지 파일 경로 설정
        img_path = f'static/img/{test_word_eng}.jpg'  

        # 이미지 불러오기
        test_image_size = 140
        test_image = cv2.imread(img_path)
        test_image = cv2.resize(test_image, (test_image_size, test_image_size))
        return test_image

    w = 640
    h = 480

    # game_random_words 고정 위치 
    word_position = [(80, 330), (115, 205), (265, 100), (435, 205), (470, 330)]

    # 글자 뒤에 배경화면
    class WhiteBox:
        def __init__(self, size_x1, size_y1, size_x2, size_y2):
            self.point_left = (size_x1, size_y1)
            self.point_rigtht = (size_x2, size_y2)

        def draw_box(self, img):
            color = (255, 255, 255)
            thickness = -1
            return cv2.rectangle(img, self.point_left, self.point_rigtht, color, thickness)

    # 타이머 설정
    class Timer:
        def __init__(self, time_limit):
            self.time_limit = time_limit
            self.start_time = None

        def start(self):
            self.start_time = time.time()

        def get_remaining_time(self):
            if self.start_time is None:
                return str("타이머를 설정하지 않았습니다")

            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.time_limit - elapsed_time)
            return remaining_time

    # 점수 관련
    CIRCLE_COLOR = (0, 102, 0)
    CIRCLE_TICKNESS = 12
    X_COLOR = (0, 0, 255)
    X_LENGTH = 100
    X_THICKNESS = 12

    # 게임 관련
    game_set = False
    game_start_timer = None
    game_end_timer = None
    game_score = 0

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, img = cap.read()
            if not ret:  # 캡쳐 오류 처리
                continue
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                # 검지 인식
                    joint = np.zeros((21, 3))
                    finger_x = int(res.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 640)
                    finger_y = int(res.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 480)

                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                                                ))
                    angle = np.degrees(angle)
                    data = np.array([angle], dtype=np.float32)
                    ret, rdata, neig, dist = knn.findNearest(data, 5)
                    idx = int(rdata[0][0])
                    #print(idx)

                    #mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 마지막에 삭제 필요
                    
                    if not game_set: # 게임 시작 전
                        if idx == 2:
                            if game_start_timer is None:  
                                game_start_timer = Timer(3)  # 타이머 3초
                                game_start_timer.start()
                                #print('게임 시작')
                            
                            remaining_time = game_start_timer.get_remaining_time()
                            time_text = f"{int(remaining_time)+1}"
                            # cv2.putText(img, time_text, (250, 280), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8, cv2.LINE_AA)
                            # cv2.circle(img, (312, 220), 140, (255, 255, 255), -1)
                            img = myPutText(img, time_text, (230, 90), FONT_SIZE+250, (0,0,0))

                            if remaining_time <= 0:
                                game_set = True
                                # print(game_set)
                        else:
                            game_start_timer = None
                            # print(game_set)

                            white_box = WhiteBox(140, 275, 505, 195)
                            img = white_box.draw_box(img)
                            img = myPutText(img, "       브이를 3초 동안 보이면\n          게임이 시작됩니다!", (75, 200), FONT_SIZE, (0, 0, 0))


                    else:  # 게임 시작 후
                        if game_end_timer is None:
                            game_end_timer = Timer(20) # 타이머 15초
                            game_end_timer.start()
                            
                        remaining_time = game_end_timer.get_remaining_time()
                        img = cv2.rectangle(img, (15, 0), (270, 35), (LOOPY), -1)
                        img = myPutText(img, f" 남은 시간 : {int(remaining_time)}초", (10, 0), FONT_SIZE, (255, 255, 255))

                        if remaining_time <= 0:
                            pygame.mixer.init()
                            pygame.mixer.music.load('sound/end.mp3')
                            pygame.mixer.music.play()

                            time.sleep(1.5)
                            
                            game_end_timer = None
                            game_set = False 
                            game_score = 0

                        game_words_position = {}
                        
                        for game_random_word, word_loc in zip(game_random_words, word_position):
                            word_size = len(game_random_word) * FONT_SIZE
                            rectangle_size = (word_size, FONT_SIZE)
                            game_words_position[game_random_word] = word_loc

                            fleft = word_loc[0]
                            fright = word_loc[1]
                            bleft = rectangle_size[0]
                            bright = rectangle_size[1]
                            rectangle_bottom_right = (fleft + bleft, fright + bright)

                            # game_random_words 배치
                            img = cv2.rectangle(img, word_loc, rectangle_bottom_right, RECTANGLE_COLOR, -1)
                            img = myPutText(img, game_random_word, word_loc, FONT_SIZE, FONT_COLOR)

                            # 점수 표시
                            img = cv2.rectangle(img, (140, 80), (210, 120), (LOOPY), -1)
                            img = myPutText(img, f" X {game_score}", (130, 80), FONT_SIZE+10, (255, 255, 255))

                            # 루피
                            loopy = cv2.imread('static/img/loopy11.png', cv2.IMREAD_UNCHANGED)
                            resized_loopy = cv2.resize(loopy, (140, 140))
                            x_offset = 0
                            y_offset = 25
                            for i in range(3):
                                img[y_offset:y_offset+resized_loopy.shape[0], x_offset:x_offset+resized_loopy.shape[1], i] = \
                                    img[y_offset:y_offset+resized_loopy.shape[0], x_offset:x_offset+resized_loopy.shape[1], i] * \
                                    (1.0 - resized_loopy[:, :, 3] / 255.0) + \
                                    resized_loopy[:, :, i] * (resized_loopy[:, :, 3] / 255.0)


                            # test_word 이미지 보여주기
                            test_image_size = 140
                            test_image = get_img()
                            img[0:test_image_size, w-test_image_size:w] = test_image
                            #print(test_word)

                            if idx == 1 and (fleft <= finger_x <= fleft + bleft) and (fright <= finger_y <= fright + bright):
                                if game_random_word == test_word:  # 정답을 맞힌 경우  
                                    cv2.circle(img, (300, 300), 150, CIRCLE_COLOR, CIRCLE_TICKNESS)

                                    pygame.mixer.init()
                                    pygame.mixer.music.load('sound/circle.mp3')
                                    pygame.mixer.music.play()
                                    
                                    game_score += 1  

                                    game_random_words = random.sample(kids_words, game_words_number)
                                    test_word = random.choice(game_random_words)

                                    time.sleep(0.4)
                    
                
                                else:  # 정답을 못 맞힌 경우
                                    cv2.line(img, (300 - X_LENGTH, 300 - X_LENGTH), (300 + X_LENGTH, 300 + X_LENGTH), X_COLOR, X_THICKNESS)
                                    cv2.line(img, (300 + X_LENGTH, 300 - X_LENGTH), (300 - X_LENGTH, 300 + X_LENGTH), X_COLOR, X_THICKNESS)
                                    pygame.mixer.init()
                                    pygame.mixer.music.load('sound/x.mp3')
                                    pygame.mixer.music.play()                            
                                     
            else:
                white_box = WhiteBox(180, 250, 470, 205)
                img = white_box.draw_box(img)
                img = myPutText(img, "손을 보여주세요 :)", (190, 210), FONT_SIZE, FONT_COLOR)                            
                                

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes() 

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@bp.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/quiz')
def hand_tracking():
    return render_template('quiz.html')