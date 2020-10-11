import face_recognition
import os
import cv2

video_capture = cv2.VideoCapture(0)

KNOWN_FACE_DIR = "E:\Tools\CodeBase\FaceRecognition"
UNKNOWN_FACE_DIR = "un_known_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"

print("Loading Known Faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACE_DIR):
    print(name)
    for filename in os.listdir(f"{KNOWN_FACE_DIR}/{name}"):
        print(filename)
        image = face_recognition.load_image_file(f"{KNOWN_FACE_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print('Completed Learning')
while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        print(results)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match Found :{match}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(frame, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(frame, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), FONT_THICKNESS)

    cv2.imshow('Webcam', frame)
    print('framed Webcam')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
