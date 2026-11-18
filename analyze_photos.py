import cv2, os, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

src = r'E:\photos ml'
CLASS_NAMES = ['ac remote','earbuds','geometry box','mobile,phone','notebook',
               'pen','scale','stapler','water bottle','whitener']

def extract(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    h, w = img.shape[:2]
    mh, mw = int(h*0.15), int(w*0.10)
    img = img[mh:h-mh, mw:w-mw]
    r = cv2.resize(img, (128,128))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)

    feats = []

    # === HIGH WEIGHT: Saturation histogram (64 bins) - separates colourful vs grey
    s_hist = cv2.calcHist([hsv],[1],None,[64],[0,256]).flatten()
    s_hist /= (s_hist.sum()+1e-7)
    feats.append(np.tile(s_hist, 5))   # repeat 5x to increase weight

    # === HIGH WEIGHT: Hue histogram (64 bins) - separates by actual colour
    h_hist = cv2.calcHist([hsv],[0],None,[64],[0,180]).flatten()
    h_hist /= (h_hist.sum()+1e-7)
    feats.append(np.tile(h_hist, 5))

    # === HIGH WEIGHT: Edge density 8x8 grid - separates phone(low) vs notebook(high)
    edges = cv2.Canny(gray, 30, 100)
    step = 128//8
    edge_grid = np.array([edges[i*step:(i+1)*step,j*step:(j+1)*step].mean()/255.0
                          for i in range(8) for j in range(8)], dtype='float32')
    feats.append(np.tile(edge_grid, 5))

    # === MEDIUM: BGR histograms
    for ch in range(3):
        hist = cv2.calcHist([r],[ch],None,[32],[0,256]).flatten()
        hist /= (hist.sum()+1e-7)
        feats.append(hist)

    # === MEDIUM: Equalised pixels (shape/texture, lighting-independent)
    gray_eq = cv2.equalizeHist(gray)
    feats.append(cv2.resize(gray_eq,(32,32)).flatten().astype('float32')/255.0)

    # === LOW: Sobel direction histograms
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sx_h, _ = np.histogram(np.clip(sx,-255,255), bins=32, range=(-255,255), density=True)
    sy_h, _ = np.histogram(np.clip(sy,-255,255), bins=32, range=(-255,255), density=True)
    feats.append(sx_h.astype('float32'))
    feats.append(sy_h.astype('float32'))

    # === Laplacian texture
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    s4 = 128//4
    lg = np.array([lap[i*s4:(i+1)*s4,j*s4:(j+1)*s4].mean() for i in range(4) for j in range(4)], dtype='float32')
    lg /= (lg.max()+1e-7)
    feats.append(lg)

    return np.concatenate(feats).astype('float32')

X, y = [], []
for cid, cname in enumerate(CLASS_NAMES):
    folder = os.path.join(src, cname)
    for fname in sorted([f for f in os.listdir(folder) if f.lower().endswith('.jpg')]):
        f = extract(os.path.join(folder, fname))
        if f is not None: X.append(f); y.append(cid)
X, y = np.array(X), np.array(y)
print(f'Feature dim: {X.shape[1]}')

rf = Pipeline([('sc',StandardScaler()),('clf',RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1))])
svc = Pipeline([('sc',StandardScaler()),('clf',SVC(C=10,kernel='rbf',probability=True,random_state=42))])

skf = StratifiedKFold(5,shuffle=True,random_state=42)
rf_scores  = cross_val_score(rf,  X, y, cv=skf)
svc_scores = cross_val_score(svc, X, y, cv=skf)
print(f'RF  cv5: {rf_scores.mean():.3f}')
print(f'SVC cv5: {svc_scores.mean():.3f}')

# Show confusion on mobile vs notebook specifically
all_pred, all_true = [], []
for tr, te in skf.split(X,y):
    rf.fit(X[tr],y[tr])
    all_pred.extend(rf.predict(X[te]))
    all_true.extend(y[te])

cm = confusion_matrix(all_true, all_pred)
# Show mobile(3) and notebook(4) rows
print("\nMobile row:  ", cm[3])
print("Notebook row:", cm[4])
print("mobile->notebook errors:", cm[3][4], "  notebook->mobile errors:", cm[4][3])
