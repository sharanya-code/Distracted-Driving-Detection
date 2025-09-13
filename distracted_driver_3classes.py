import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2
import timm
import statistics
import numpy as np
from collections import deque
import torch.nn.functional as F

# 0. Configuration
NUM_CLASSES = 3  # 0: normal driving, 1: texting, 2: talking on phone
TYPE_MAP = {
    'c0': 0,  # normal driving
    'c1': 1,  # texting (right)
    'c2': 2,  # talking (right)
    'c3': 1,  # texting (left)
    'c4': 2,  # talking (left)
}
LABEL_MAP = {
    0: 'normal driving',
    1: 'texting',
    2: 'talking on phone',
}

# 1. Dataset Definitions
class DistractedDriverDataset(Dataset):
    """
    Labeled dataset. Expects subfolders 'c0'..'c4' under root_dir.
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        for cls in sorted(os.listdir(root_dir)):
            if cls not in TYPE_MAP:
                continue
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    label = TYPE_MAP[cls]
                    self.samples.append((os.path.join(cls_dir, fn), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class DistractedDriverTestDataset(Dataset):
    """
    Unlabeled test dataset. Returns (image, filename).
    """
    def __init__(self, test_dir, transform=None):
        self.items = [
            os.path.join(test_dir, fn)
            for fn in sorted(os.listdir(test_dir))
            if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]
        img  = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(path)

# 2. Model Definition

def create_model(backbone='mobilenetv3_small_100', num_classes=NUM_CLASSES):
    model = timm.create_model(backbone, pretrained=True)
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError(f'Unsupported backbone: {backbone}')
    return model

# 3. Training Loop
def train(data_dir, epochs, batch_size, lr, val_split, backbone):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    ds = DistractedDriverDataset(data_dir, transform)
    val_size   = int(len(ds) * val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = create_model(backbone=backbone).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        correct, total = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
        train_acc = correct / total
        print(f"[{backbone}] Epoch {epoch}/{epochs} — Train Acc: {train_acc:.4f}", end='')

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)
        val_acc = val_correct / val_total
        print(f"  Val Acc: {val_acc:.4f}", end='')

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  LR: {lr_now:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = f"{backbone}_best.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  → Saved best model to {ckpt}")

    print(f"[{backbone}] Training complete. Best Val Acc: {best_acc:.4f}\n")

# 4. Inference Functions

def inference_webcam(model_path, backbone='mobilenetv3_small_100'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = create_model(backbone=backbone).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    events, current, start_ts = [], None, None
    while True:
        ret, frame = cap.read()
        if not ret: break
        ts = time.time()
        inp = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad(): cls = model(inp).argmax(dim=1).item()
        if cls != current:
            if current is not None:
                events.append({'class': current, 'start': start_ts, 'end': ts})
            current, start_ts = cls, ts
        cv2.putText(frame, LABEL_MAP[cls], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1)==27: break
    if current is not None:
        events.append({'class': current, 'start': start_ts, 'end': time.time()})
    with open('drive_events.json','w') as f: json.dump(events,f,indent=2)
    cap.release(); cv2.destroyAllWindows()


def inference_testset(model_path, backbone, test_dir, out_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = create_model(backbone=backbone).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    ds = DistractedDriverTestDataset(test_dir, transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    rows=[]
    with torch.no_grad():
        for imgs, fnames in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            rows += [{'img':f,'class':p} for f,p in zip(fnames,preds)]
    import csv
    with open(out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,['img','class']); w.writeheader(); w.writerows(rows)
    print(f"Saved {out_csv}")


def ensemble_testset(model_paths, backbones, test_dir, out_csv):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models=[]
    for mp,bb in zip(model_paths,backbones):
        m=create_model(bb).to(device)
        m.load_state_dict(torch.load(mp,map_location=device))
        m.eval(); models.append(m)
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds=DistractedDriverTestDataset(test_dir,transform)
    loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=4)
    rows=[]
    with torch.no_grad():
        for imgs,fnames in loader:
            imgs=imgs.to(device)
            logits_sum=None
            for m in models:
                out=m(imgs)
                logits_sum=out if logits_sum is None else logits_sum+out
            preds=logits_sum.argmax(dim=1).cpu().tolist()
            rows+= [{'img':f,'class':p} for f,p in zip(fnames,preds)]
    import csv
    with open(out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,['img','class']); w.writeheader(); w.writerows(rows)
    print(f"Saved ensemble CSV to {out_csv}")


def ensemble_cam(model_paths, backbones, cam_index=1, log_path='drive_events_ensemble.json'):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models=[]
    for mp,bb in zip(model_paths,backbones):
        m=create_model(bb).to(device)
        m.load_state_dict(torch.load(mp,map_location=device))
        m.eval(); models.append(m)
    with open(log_path,'w') as f: json.dump([],f)
    cap=cv2.VideoCapture(cam_index)
    current,start_ts=None,None
    preprocess=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    while True:
        ret,frame=cap.read()
        if not ret: break
        ts=time.time()
        inp=preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_sum=None
            for m in models:
                out=m(inp); logits_sum=out if logits_sum is None else logits_sum+out
            cls=logits_sum.argmax(dim=1).item()
        if cls!=current:
            if current is not None:
                with open(log_path,'r+') as f:
                    data=json.load(f); data.append({'class':current,'start':start_ts,'end':ts})
                    f.seek(0); json.dump(data,f,indent=2); f.truncate()
            current,start_ts=cls,ts
        cv2.putText(frame,LABEL_MAP[cls],(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Ensemble',frame)
        if cv2.waitKey(1)==27: break
    if current is not None:
        with open(log_path,'r+') as f:
            data=json.load(f); data.append({'class':current,'start':start_ts,'end':time.time()})
            f.seek(0); json.dump(data,f,indent=2); f.truncate()
    cap.release(); cv2.destroyAllWindows()


def annotate_video(model_paths, backbones, video_path, output_path='annotated_video.mp4', buffer_size=5, conf_thresh=0.7):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models=[]
    for mp,bb in zip(model_paths,backbones):
        m=create_model(bb).to(device)
        raw=torch.load(mp,map_location=device)
        sd=raw.get('state_dict',raw)
        m.load_state_dict(sd,strict=False); m.eval(); models.append(m)
    preprocess=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    buf=deque(maxlen=buffer_size)
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open {video_path}")
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out=cv2.VideoWriter(output_path,fourcc,fps,(w,h))
    while True:
        ret,frame=cap.read();
        if not ret: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        eq=cv2.equalizeHist(gray)
        kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)
        sharp=cv2.filter2D(eq,-1,kernel)
        proc=cv2.merge([sharp,sharp,sharp])
        inp=preprocess(proc).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_sum=None
            for m in models: out_logits=m(inp); logits_sum=out_logits if logits_sum is None else logits_sum+out_logits
            probs=F.softmax(logits_sum,dim=1); max_prob,cls=probs.max(dim=1)
            cls=int(cls.item()); max_prob=float(max_prob.item())
        if max_prob<conf_thresh and buf: cls=buf[-1]
        buf.append(cls)
        try: smooth_cls=statistics.mode(buf)
        except statistics.StatisticsError: smooth_cls=cls
        label=f"{LABEL_MAP[smooth_cls]} ({max_prob:.2f})"
        cv2.putText(frame,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        out.write(frame)
    cap.release(); out.release()
    print(f"Saved annotated video to {output_path}")

# 5. Main
if __name__ == '__main__':
    mode = 'annotate_video'  # options: train, inference, test, ensemble_test, ensemble_cam, annotate_video
    data_dir    = 'path/to/train'
    test_dir    = 'path/to/test'
    backbones   = ['mobilenetv3_small_100', 'efficientnet_b0']
    model_paths = ['mobilenetv3_small_100_best.pth', 'efficientnet_b0_best.pth']
    out_csv     = 'submission.csv'
    epochs      = 15
    batch_size  = 32
    lr          = 1e-4
    val_split   = 0.2
    video_path  = 'path/to/input.mp4'
    
    if mode == 'train_ensemble':
        for bb in backbones:
            print(f"\n=== Training backbone: {bb} ===")
            train(data_dir, epochs, batch_size, lr, val_split, backbone=bb)
        print(f"\n=== Running ensemble on test set ===")
        model_paths = [f'{bb}_best.pth' for bb in backbones]
        ensemble_testset(model_paths, backbones, test_dir, out_csv)
        print(f"\nEnsemble results saved to {out_csv}")
    elif mode == 'train':
        train(data_dir, epochs, batch_size, lr, val_split, backbones[0])
    elif mode == 'inference':
        inference_webcam(model_paths[0], backbones[0])
    elif mode == 'test':
        inference_testset(model_paths[0], backbones[0], test_dir, out_csv)
    elif mode == 'ensemble_test':
        ensemble_testset(model_paths, backbones, test_dir, out_csv)
    elif mode == 'ensemble_cam':
        ensemble_cam(model_paths, backbones)
    elif mode == 'annotate_video':
        annotate_video(model_paths, backbones, video_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
