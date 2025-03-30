import cv2
import mediapipe as mp
import pandas as pd

#ตั้งค่า Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

#ชื่อคำที่ต้องการเก็บข้อมูล
label_names = ["1","2","3"]
data = []
labels = []

cap = cv2.VideoCapture(0)

for label_index, label_name in enumerate(label_names):
    print(f" ทำท่า '{label_name}' แล้วกด 's' เพื่อบันทึก หรือ 'q' เพื่อข้าม")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ดึงค่าตำแหน่งมือ
                landmark_list = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
                data.append(landmark_list)
                labels.append(label_index)

        cv2.imshow("Collect Data", image)
        key = cv2.waitKey(1)
        if key == ord('s'):
            print(f"📸 บันทึกท่า '{label_name}' สำเร็จ")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#  บันทึกข้อมูลลง CSV
df = pd.DataFrame(data)
df["label"] = labels
df.to_csv("sign_language_data.csv", index=False)
print(" ข้อมูลถูกบันทึกลงแล้ว")

