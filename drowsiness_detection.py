import face_recognition
import cv2
import time
import playsound
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread

MIN_AER = 0.30
EYE_AR_CONSEC_FRAME = 10
COUNTER = 0
ALARM_ON = False

def sound_alarm(soundfile):
    playsound.playsound(soundfile)

def eye_aspect_ratio(eye):
    eyePointA = dist.euclidean(eye[1], eye[5])
    eyePointB = dist.euclidean(eye[2], eye[4])
    eyePointC = dist.euclidean(eye[0], eye[3])
    ear = (eyePointA + eyePointB) / (2.0 * eyePointC)
    return ear

def main():
    global COUNTER, ALARM_ON
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 320)  # Set the resolution - adjust if needed
    video_capture.set(4, 240)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # If the frame is not successfully captured, exit the loop
        
        face_landmarks_list = face_recognition.face_landmarks(frame)
        for face_landmark in face_landmarks_list:
            leftEye = face_landmark['left_eye']
            rightEye = face_landmark['right_eye']
            leftEar = eye_aspect_ratio(leftEye)
            rightEar = eye_aspect_ratio(rightEye)
            ear = ((leftEar + rightEar) / 2)

            lpst = np.array(leftEye)
            rpst = np.array(rightEye)

            cv2.polylines(frame, [lpst], True, (255, 255, 0), 1)
            cv2.polylines(frame, [rpst], True, (255, 255, 0), 1)
            
            if ear < MIN_AER:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAME:
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=sound_alarm, args=('drowsiness_alarm.wav',))
                        t.daemon = True
                        t.start()
                cv2.putText(frame, "Alert: Drowsiness Detected!", (5, 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 0), 1)
            else:
                COUNTER = 0
                ALARM_ON = False
            cv2.putText(frame, f"EAR: {ear:.2f}", (250, 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 0), 1)
            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    # Release the video capture and destroy all windows after the loop
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()