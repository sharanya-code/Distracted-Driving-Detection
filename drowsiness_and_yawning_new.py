import argparse
import os
import time
import json
from threading import Thread
from collections import deque

import cv2
import imutils
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# python drowsiness_and_yawning_new.py --webcam 1

alarm_status  = False
alarm_status2 = False
saying        = False

drowsiness_events = []
yawn_events = []
current_drowsy = None
current_yawn = None
log_file = "drowsiness_yawn_log.json"

def alarm(msg):
    global alarm_status, alarm_status2, saying
    while alarm_status or alarm_status2:
        if not saying:
            saying = True
            os.system(f'espeak "{msg}"')
            saying = False

def save_log():
    log = {"drowsiness_events": drowsiness_events, "yawn_events": yawn_events}
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_ear(pts, eye_idxs):
    eye = pts[eye_idxs]
    return eye_aspect_ratio(eye), eye

def get_lip_distance(pts):

    pairs = [(13, 14), (78, 308), (61, 291)]
    dists = [abs(pts[up][1] - pts[low][1]) for up, low in pairs]
    return np.mean(dists)

def main():
    global alarm_status, alarm_status2, current_drowsy, current_yawn

    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="Index of webcam (default: 0)")
    args = ap.parse_args()

    EYE_AR_THRESH        = 0.3
    EYE_AR_CONSEC_FRAMES = 30
    YAWN_THRESH          = 7.5 
    YAWN_MIN_FRAMES      = 15    

    counter = 0
    yawn_counter = 0

    mp_face   = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    LEFT_EYE_IDXS  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

    print("-> Starting video stream...")
    vs = cv2.VideoCapture(args.webcam)
    time.sleep(1.0)

    yawn_buffer = deque(maxlen=10)

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=450)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            lm = results.multi_face_landmarks[0].landmark
            pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm])

            leftEAR, leftEye  = get_ear(pts, LEFT_EYE_IDXS)
            rightEAR, rightEye = get_ear(pts, RIGHT_EYE_IDXS)
            ear = (leftEAR + rightEAR) / 2.0

            lip_dist = get_lip_distance(pts)
            yawn_buffer.append(lip_dist)
            smoothed_lip = np.mean(yawn_buffer)

            cv2.polylines(frame, [leftEye],  True, (0,255,0), 1)
            cv2.polylines(frame, [rightEye], True, (0,255,0), 1)

            ts = time.time()

            if ear < EYE_AR_THRESH:
                counter += 1
                if counter >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        current_drowsy = {"start": ts}
                        Thread(target=alarm, args=("Wake up, sir!",), daemon=True).start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                if alarm_status and current_drowsy:
                    current_drowsy["end"] = ts
                    drowsiness_events.append(current_drowsy)
                    save_log()
                    current_drowsy = None
                counter = 0
                alarm_status = False

            if smoothed_lip > YAWN_THRESH:
                yawn_counter += 1
                if yawn_counter >= YAWN_MIN_FRAMES:
                    if not alarm_status2:
                        alarm_status2 = True
                        current_yawn = {"start": ts}
                        Thread(target=alarm, args=("Take some fresh air, sir!",), daemon=True).start()
                    cv2.putText(frame, "YAWN ALERT!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                if alarm_status2 and current_yawn:
                    current_yawn["end"] = ts
                    yawn_events.append(current_yawn)
                    save_log()
                    current_yawn = None
                alarm_status2 = False
                yawn_counter = 0

            cv2.putText(frame, f"EAR: {ear:.2f}",  (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, f"LIP: {smoothed_lip:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    ts = time.time()
    if current_drowsy:
        current_drowsy["end"] = ts
        drowsiness_events.append(current_drowsy)
        save_log()
    if current_yawn:
        current_yawn["end"] = ts
        yawn_events.append(current_yawn)
        save_log()

    vs.release()
    cv2.destroyAllWindows()
    print(f"-> Log saved to {log_file}")

if __name__ == "__main__":
    main()


