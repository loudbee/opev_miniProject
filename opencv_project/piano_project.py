import cv2
import mediapipe as mp
import math
import pygame

# 초기화
pygame.mixer.init()

# 음표와 해당 소리 파일 경로를 매핑
note_sounds = {
    "C4": "opencv_project/sounds/C4.mp3",
    "D4": "opencv_project/sounds/D4.mp3",
    "E4": "opencv_project/sounds/E4.mp3",
    "F4": "opencv_project/sounds/F4.mp3",
    "G4": "opencv_project/sounds/G4.mp3",
    "A4": "opencv_project/sounds/A4.mp3",
    "B4": "opencv_project/sounds/B4.mp3",
    "C5": "opencv_project/sounds/C5.mp3",
}

# 피아노 건반 키 좌표
piano_keys = {
    "C4": (300, 600),
    "D4": (400, 600),
    "E4": (500, 600),
    "F4": (600, 600),
    "G4": (700, 600),
    "A4": (800, 600),
    "B4": (900, 600),
    "C5": (1000, 600),
}

# 피아노 건반 눌림 여부를 저장할 딕셔너리 초기화
key_pressed = {key: False for key in piano_keys}

# MediaPipe Hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 웹캠 관련 설정
cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)  # 웹캠을 사용하여 손 검출을 시도합니다.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# UI 관련 설정(기본값)
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# font_color = (255, 255, 255)
# font_thickness = 2

while True:
    ret, frame = cap.read()  # 프레임 읽기

    if not ret:
        break

    # BGR 이미지를 RGB 이미지로 변환
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 피아노 건반 그리기
    for key, (key_x, key_y) in piano_keys.items():
        cv2.rectangle(
            frame, (key_x - 50, key_y - 50), (key_x + 50, key_y + 50), (255, 255, 0), 2
        )

        # 피아노건반 이름 표시
        cv2.putText(
            frame,
            key,
            (key_x - 10, key_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )

    # 손 검출 수행
    results = hands.process(frame_rgb)

    # 검출된 손 표시
    if results.multi_hand_landmarks:
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            # 손가락에 점과 손가락 끝점을 인식하기위한 리스트 초기화
            landmark_points = []

            for point_idx, point in enumerate(landmarks.landmark):
                # point.x,y로 받은 좌표값을 현제 프레임의 가로,너비로 곱해서 정확한 위치 찾기
                x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                landmark_points.append((x, y))

                # 손 관절 점 그리기
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # 손가락 끝지점 가져오기
            thumb_tip = (landmark_points[4][0], landmark_points[4][1])
            index_tip = (landmark_points[8][0], landmark_points[8][1])
            middle_tip = (landmark_points[12][0], landmark_points[12][1])
            ring_tip = (landmark_points[16][0], landmark_points[16][1])
            little_tip = (landmark_points[20][0], landmark_points[20][1])

            # 피아노 건반 눌림 여부 판단
            for key, (key_x, key_y) in piano_keys.items():
                # 피타고라스의 정리(두점 사이의 거리를 재기위한것)
                distance_thumb = math.sqrt(
                    (thumb_tip[0] - key_x) ** 2 + (thumb_tip[1] - key_y) ** 2
                )

                distance_index = math.sqrt(
                    (index_tip[0] - key_x) ** 2 + (index_tip[1] - key_y) ** 2
                )

                distance_middle = math.sqrt(
                    (middle_tip[0] - key_x) ** 2 + (middle_tip[1] - key_y) ** 2
                )

                distance_ring = math.sqrt(
                    (ring_tip[0] - key_x) ** 2 + (ring_tip[1] - key_y) ** 2
                )

                distance_little = math.sqrt(
                    (little_tip[0] - key_x) ** 2 + (little_tip[1] - key_y) ** 2
                )

                # 손가락이 건반을 누르면 소리를 재생하고 건반 색깔 변화 & 오른쪽위 텍스트 표시
                if (
                    # any는 괄호안에 있는것이 하나라도 참이면 참이다.
                    any(
                        distance < 50
                        for distance in [
                            distance_thumb,
                            distance_index,
                            distance_middle,
                            distance_ring,
                            distance_little,
                        ]
                    )
                    and not key_pressed[key]
                ):
                    key_pressed[key] = True

                    # 피아노 건반 눌림 표시
                    cv2.rectangle(
                        frame,
                        (key_x - 50, key_y - 50),
                        (key_x + 50, key_y + 50),
                        (255, 255, 255),
                        -1,
                    )

                    # 피아노 건반 텍스트 지워짐 방지
                    cv2.putText(
                        frame,
                        key,
                        (key_x - 10, key_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )

                    # 오른쪽 위 누른 키 표시
                    cv2.putText(
                        frame,
                        key,
                        (1000, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        5,
                        (255, 255, 0),
                        5,
                    )

                    # 개발중 어떤 키가 눌렸는지 확인하고 if문 안에 들어왔는지 확인용도
                    print(f"Pressing key {key}")
                    pygame.mixer.music.load(note_sounds[key])
                    pygame.mixer.music.play()
                # 손가락이 건반에서 손을 떼면 상태 초기화
                elif (
                    all(
                        distance >= 50
                        for distance in [
                            distance_thumb,
                            distance_index,
                            distance_middle,
                            distance_ring,
                            distance_little,
                        ]
                    )
                    and key_pressed[key]
                ):
                    key_pressed[key] = False

    # 화면에 출력
    cv2.imshow("Piano", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()
