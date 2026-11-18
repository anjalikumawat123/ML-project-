import cv2, os, numpy as np, sys
sys.path.insert(0, r'E:\photos ml\project')
from data_preparation import extract_features, CLASS_NAMES, _find_object_crop
import joblib

src = r'E:\photos ml'
MODELS_DIR = r'E:\photos ml\project\models'

rf  = joblib.load(os.path.join(MODELS_DIR, 'Random_Forest.pkl'))
svc = joblib.load(os.path.join(MODELS_DIR, 'Support_Vector_Classifier.pkl'))
lr  = joblib.load(os.path.join(MODELS_DIR, 'Logistic_Regression.pkl'))

print('Testing all training images through auto-crop pipeline...\n')
correct_rf = correct_svc = correct_lr = total = 0
errors = []

for cid, cname in enumerate(CLASS_NAMES):
    folder = os.path.join(src, cname)
    imgs = sorted([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])
    cc_rf = cc_svc = cc_lr = 0
    for fname in imgs:
        img = cv2.imread(os.path.join(folder, fname))
        feat = extract_features(img, augment=False).reshape(1, -1)
        total += 1
        prf  = rf.predict(feat)[0]
        psvc = svc.predict(feat)[0]
        plr  = lr.predict(feat)[0]
        if prf  == cid: correct_rf  += 1; cc_rf  += 1
        else: errors.append((cname, CLASS_NAMES[prf], fname))
        if psvc == cid: correct_svc += 1; cc_svc += 1
        if plr  == cid: correct_lr  += 1; cc_lr  += 1
    n = len(imgs)
    print(f'  {cname:<20}  RF={cc_rf}/{n}  SVC={cc_svc}/{n}  LR={cc_lr}/{n}')

print()
print(f'Overall on ALL training images:')
print(f'  RF : {correct_rf}/{total} = {correct_rf/total*100:.1f}%')
print(f'  SVC: {correct_svc}/{total} = {correct_svc/total*100:.1f}%')
print(f'  LR : {correct_lr}/{total} = {correct_lr/total*100:.1f}%')

print()
print('RF errors (true -> predicted):')
for true_cls, pred_cls, fname in errors[:20]:
    print(f'  {true_cls:<20} -> {pred_cls:<20} {fname[:35]}')

# Now show what the auto-crop actually produces for each class
print()
print('Auto-crop effectiveness per class:')
print('(crop_w x crop_h vs original w x h, % object area)')
for cname in CLASS_NAMES:
    folder = os.path.join(src, cname)
    fname = sorted([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])[0]
    img = cv2.imread(os.path.join(folder, fname))
    oh, ow = img.shape[:2]
    crop = _find_object_crop(img)
    ch, cw = crop.shape[:2]
    pct = cw * ch * 100 / (ow * oh)
    print(f'  {cname:<20}  full={ow}x{oh}  crop={cw}x{ch}  ({pct:.0f}% of frame)')
