import cv2
import mediapipe as mp
import numpy as np
import joblib  # โหลดโมเดล

# ✅ โหลดโมเดล KNN
model = joblib.load("hand_sign_model.pkl")
label_names = ["1","2","3"]

# ✅ ตั้งค่า Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ✅ ดึงค่าตำแหน่งมือ
            landmark_list = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]

            # ✅ ทำนายผลลัพธ์
            if len(landmark_list) == model.n_features_in_:
                prediction = model.predict([landmark_list])
                predicted_text = label_names[prediction[0]]
                cv2.putText(image, predicted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Sign Detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
