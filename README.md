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
- **EYE_AR_THRESH = 0.3** → threshold for eye closure.
- **EYE_AR_CONSEC_FRAMES = 30** → number of frames before drowsiness triggers.
- **YAWN_THRESH = 7.5** → lip distance threshold for yawning.
- **YAWN_MIN_FRAMES = 15** → frames required to confirm yawning.


## 🎬 Demo

https://github.com/user-attachments/assets/b4b56956-a4ec-4ce1-9eb2-fad387e6d062


# Distracted Driver Detection 📱🚗  

This module trains and evaluates deep learning models to detect **distracted driving behaviors**. The system is designed to recognize:  
- **Normal Driving (class 0)**  
- **Texting (class 1)**  
- **Talking on the Phone (class 2)**  

By automating the detection of these behaviors, the system supports **driver safety research**, fleet monitoring, and development of **driver assistance systems**.  

---

## 🔹 Dataset  

The script expects a dataset organized into class-specific subfolders under a given `root_dir`.  

### Training Data Format:  

- Classes `c1` and `c3` are **merged** into class 1 (texting).  
- Classes `c2` and `c4` are **merged** into class 2 (talking).  
- The model therefore predicts **3 consolidated classes**.  

### Test Data Format:  

Unlabeled test data is supported for inference and CSV output.  

---

## 🔹 Model  

This script uses **transfer learning** with [timm](https://github.com/rwightman/pytorch-image-models).  
- Default: `mobilenetv3_small_100`  
- Other options: `efficientnet_b0`, etc.  

### Architecture  
- Pretrained backbone (ImageNet weights).  
- Final classifier layer replaced with a new `nn.Linear` → outputs **3 classes**.  

---

## 🔹 Training  

The training loop is defined in the `train()` function.  

1. **Preprocessing**  
   - Grayscale conversion → 3 channels.  
   - Resize to `224 × 224`.  
   - Random horizontal flip (augmentation).  
   - Normalize with ImageNet mean/std.  

2. **Split**  
   - Training/validation split controlled by `val_split` (default: 0.2).  

3. **Optimization**  
   - Optimizer: `Adam` with weight decay `1e-5`.  
   - Loss: `CrossEntropyLoss`.  
   - Scheduler: `CosineAnnealingLR`.  

4. **Checkpoints**  
   - Saves the **best model** by validation accuracy to `<backbone>_best.pth`.  

## 🔹 Inference
The trained models can be used for:

- Webcam Inference: Real-time detection with labels overlayed.

- Test Set Evaluation: Generates CSV predictions.

- Video Annotation: Saves annotated videos with predicted labels and confidence.

- Ensemble Mode: Combines predictions from multiple backbones.

## 🎬 Demo





https://github.com/user-attachments/assets/2adbd14b-a7fe-4357-a971-5e24be3d2c92























