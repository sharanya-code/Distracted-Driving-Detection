# Drowsiness & Yawn Detection 😴🛑  

A real-time computer vision system to detect **driver drowsiness** and **yawning** using **MediaPipe FaceMesh**, Eye Aspect Ratio (EAR), and Lip Distance metrics.  
This project is designed with **driver safety** in mind: prolonged drowsiness and yawning can significantly increase the risk of accidents. By monitoring these behaviors, the system provides **early warnings** and logs events that can help analyze driver fatigue patterns.  

---

## 🚗 Why This Project Matters  
Driver fatigue is one of the leading causes of road accidents worldwide. Even a few seconds of inattention can be dangerous at high speeds.  
This system aims to:  
- **Alert drivers in real-time** if they show signs of drowsiness or yawning.  
- **Encourage safer driving habits** by reducing the risks of microsleeps and fatigue-related distractions.  
- **Log critical events** (`drowsiness_yawn_log.json`) for later analysis, which can be useful in:  
  - Research on driver fatigue.  
  - Fleet management (monitoring commercial drivers).  
  - Developing advanced driver-assistance systems (ADAS).  

---

## 🔹 Features  
- **Eye Closure Detection (EAR):** Identifies prolonged eye closure using a threshold.  
- **Yawning Detection (Lip Distance):** Detects yawning based on mouth opening ratio.  
- **MediaPipe FaceMesh:** Provides robust facial landmarks.  
- **Auditory Alerts (espeak):**  
  - `"Wake up, sir!"` for drowsiness.  
  - `"Take some fresh air, sir!"` for yawning.  
- **Event Logging:** Saves start/end times of events in `drowsiness_yawn_log.json`.  
- **Real-Time Visualization:** Displays EAR, lip distance, and alerts on webcam feed.  

---

## ⚙️ Parameters
-**EYE_AR_THRESH = 0.3** → threshold for eye closure.
-**EYE_AR_CONSEC_FRAMES = 30** → number of frames before drowsiness triggers.
-**YAWN_THRESH = 7.5** → lip distance threshold for yawning.
-**YAWN_MIN_FRAMES = 15** → frames required to confirm yawning.
