import cv2
import os

# Load trained model
model_path = "../model/trained_model.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

label_names = {
    1:"User_1",
    2:"User_2",
    3:"User_3",
    4:"User_4",
    5:"User_5",
    6:"User_6",
    7:"User_7", 
    8:"User_8",
    9:"User_9"
}

cap = cv2.VideoCapture(0)

print("STarting real time face recognition... press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # predict label
        label, confidence = recognizer.predict(roi_gray)
        
        name = label_names.get(label, f"User {label}")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
        cv2.putText(frame, f"{name}({int(confidence)})", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    cv2.imshow("Face Recognintion", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()