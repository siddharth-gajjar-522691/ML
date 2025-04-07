'''
This Code is developed to gather the data for training and testing purposes.
The code will turn the camera On take a snapshot and detect the faces. 
The detected faces will be storen in a saperate folder by assigning the user name and ids.

The code is working tested on April 7, 2025 2:24AM
'''


import cv2
import os

# Parameters
user_id = input("Enter ID:")
save_path = f"../data/faces/user_{user_id}"
os.makedirs(save_path, exist_ok=True)

# Load the default webc1ame
cap = cv2.VideoCapture(0)
 
#  Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
max_images = 50  #number of images collected

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect face 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        face_img = frame[y:y+h, x:x+w] 
        image_path = os.path.join(save_path, f"user_{user_id}_{count}.jpg")
        cv2.imwrite(image_path, face_img)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Dataset Collection - Press Q to quit', frame)
    
    if count >= max_images:
        print(f"Capture {max_images} images.")
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cap.release()
    cv2.destroyAllWindows()

        

