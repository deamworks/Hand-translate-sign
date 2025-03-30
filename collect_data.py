import cv2
import mediapipe as mp
import pandas as pd
import os
#hand_sign_model.pkl
#sign_language_data.csv


# ตั้งค่า Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# โหลดข้อมูลเดิม ถ้ามีไฟล์อยู่
csv_file = "sign_language_data.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    data = df.iloc[:, :-1].values.tolist()
    labels = df.iloc[:, -1].values.tolist()
    print(f"📂 พบข้อมูลเก่า {len(labels)} รายการ กำลังเพิ่มข้อมูลใหม่...")
else:
    data, labels = [], []

# ชื่อคำที่ต้องการเก็บข้อมูลใหม่
label_names = ["4","5"]

cap = cv2.VideoCapture(0)

for label_index, label_name in enumerate(label_names, start=max(labels, default=-1) + 1):
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

                # ✅ ดึงค่าตำแหน่งมือ
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

#  บันทึกข้อมูลลง CSV โดยไม่ลบของเก่า
df_new = pd.DataFrame(data)
df_new["label"] = labels
df_new.to_csv(csv_file, index=False)
print(f" ข้อมูลทั้งหมดถูกบันทึกลง {csv_file} แล้ว!")

