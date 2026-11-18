"""
webcam_capture.py  —  capture webcam frame, show crop, diagnose prediction
Press SPACE to capture. Q to quit.
"""
import cv2, os, numpy as np, sys
sys.path.insert(0, r'E:\photos ml\project')
from data_preparation import extract_features, CLASS_NAMES, _find_object_crop
import joblib

MODELS_DIR = r'E:\photos ml\project\models'
src        = r'E:\photos ml'
os.makedirs('debug_crops', exist_ok=True)

rf  = joblib.load(os.path.join(MODELS_DIR, 'Random_Forest.pkl'))
svc = joblib.load(os.path.join(MODELS_DIR, 'Support_Vector_Classifier.pkl'))
lr  = joblib.load(os.path.join(MODELS_DIR, 'Logistic_Regression.pkl'))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Hold object to camera. SPACE=capture  Q=quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Show live crop detection
    display = frame.copy()
    h, w = frame.shape[:2]
    corner_size = min(h, w) // 8
    corners = [frame[:corner_size,:corner_size], frame[:corner_size,w-corner_size:],
               frame[h-corner_size:,:corner_size], frame[h-corner_size:,w-corner_size:]]
    bg = np.median(np.vstack([c.reshape(-1,3) for c in corners]), axis=0).astype(np.float32)
    diff = np.abs(frame.astype(np.float32) - bg).mean(axis=2)
    thresh = max(18, diff.std() * 0.8)
    mask = (diff > thresh).astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) > h*w*0.03:
            x, y, bw, bh = cv2.boundingRect(largest)
            px2, py2 = int(bw*0.08), int(bh*0.08)
            bx1 = max(0,x-px2); by1 = max(0,y-py2)
            bx2 = min(w,x+bw+px2); by2 = min(h,y+bh+py2)
            cv2.rectangle(display, (bx1,by1), (bx2,by2), (0,220,80), 2)

    cv2.putText(display, "SPACE=capture  Q=quit", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,180,255), 2)
    cv2.imshow("Webcam Diagnostic", display)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27): break

    if key == ord(' '):
        cv2.imwrite('debug_crops/webcam_full.jpg', frame)
        crop = _find_object_crop(frame)
        cv2.imwrite('debug_crops/webcam_crop.jpg', crop)

        feat = extract_features(frame, augment=False).reshape(1,-1)
        r128 = cv2.resize(crop, (128,128))
        hsv  = cv2.cvtColor(r128, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(r128, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        print()
        print("=== CAPTURE ===")
        print(f"  Frame size : {frame.shape[1]}x{frame.shape[0]}")
        print(f"  Crop size  : {crop.shape[1]}x{crop.shape[0]}")
        print(f"  sat={hsv[:,:,1].mean():.1f}  hue={hsv[:,:,0].mean():.1f}  edges={edges.mean():.1f}")
        print()
        print("  Model predictions:")
        for name, model in [("Random Forest",rf),("SVC",svc),("Logistic Reg",lr)]:
            pred = model.predict(feat)[0]
            prob = model.predict_proba(feat)[0] if hasattr(model,'predict_proba') else None
            conf = prob[pred]*100 if prob is not None else 0
            print(f"    {name:<20}: {CLASS_NAMES[pred]:<20} {conf:.1f}%")
        print()
        print("  Training class stats (sat / hue / edges):")
        for obj in CLASS_NAMES:
            path = os.path.join(src, obj)
            imgs = sorted([f for f in os.listdir(path) if f.lower().endswith('.jpg')])[:10]
            sats,hues,edgs = [],[],[]
            for fn in imgs:
                im = cv2.imread(os.path.join(path,fn))
                cr = _find_object_crop(im)
                cr128 = cv2.resize(cr,(128,128))
                h2 = cv2.cvtColor(cr128, cv2.COLOR_BGR2HSV)
                g2 = cv2.cvtColor(cr128, cv2.COLOR_BGR2GRAY)
                e2 = cv2.Canny(g2,30,100)
                sats.append(h2[:,:,1].mean()); hues.append(h2[:,:,0].mean()); edgs.append(e2.mean())
            print(f"    {obj:<20} sat={np.mean(sats):5.1f}  hue={np.mean(hues):5.1f}  edges={np.mean(edgs):5.1f}")
        print()
        print("  Saved: debug_crops/webcam_full.jpg  debug_crops/webcam_crop.jpg")

cap.release()
cv2.destroyAllWindows()
