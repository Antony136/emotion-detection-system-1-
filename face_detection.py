import cv2
import numpy as np
from tensorflow.keras.models import load_model


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


model = load_model('emotion_model_49epochs.h5')


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
     
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized / 255.0

      
        roi_input = np.expand_dims(roi_normalized, axis=0)
        roi_input = np.expand_dims(roi_input, axis=-1)


        prediction = model.predict(roi_input)
        label = emotion_labels[np.argmax(prediction)]

       
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
