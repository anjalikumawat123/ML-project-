"""
webcam_test.py
===============
Smart Object Identifier — press SPACE to scan, see all 6 algorithms + properties.

HOW IT WORKS:
  - Opens webcam in SCANNING mode (live feed with targeting box)
  - Press SPACE  -> freezes frame, runs all 6 algorithms, shows results
  - Press SPACE or R  -> back to live scanning
  - Press Q / ESC  -> quit

Usage:
    python webcam_test.py
"""

import os
import sys
import cv2
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from data_preparation import extract_features, CLASS_NAMES as _DP_CLASS_NAMES

# ─── Config ───────────────────────────────────────────────────────────────────
MODELS_DIR = r"E:\photos ml\project\models"

# Import CLASS_NAMES from data_preparation so it's always in sync
CLASS_NAMES = _DP_CLASS_NAMES

# Full algorithm names + their .pkl file (order = display order)
MODEL_FILES = [
    ("Logistic Regression",       "Logistic_Regression.pkl"),
    ("Decision Tree",             "Decision_Tree.pkl"),
    ("Naive Bayes",               "Naive_Bayes.pkl"),
    ("K-Nearest Neighbours",      "K-Nearest_Neighbours.pkl"),
    ("Random Forest",             "Random_Forest.pkl"),
    ("Support Vector Classifier", "Support_Vector_Classifier.pkl"),
]

# ─── Object Properties ────────────────────────────────────────────────────────
# Keys must exactly match CLASS_NAMES entries
OBJECT_PROPERTIES = {
    "ac remote": [
        ("Category",   "Electronics / Control"),
        ("Used For",   "Controlling Air Conditioner"),
        ("Signal",     "Infrared (IR)"),
        ("Battery",    "AAA 1.5V"),
        ("Fun Fact",   "IR signal travels at speed of light"),
    ],
    "earbuds": [
        ("Category",   "Electronics / Audio"),
        ("Used For",   "Listening to music, calls"),
        ("Material",   "Plastic, Silicone, Metal"),
        ("Connects",   "Bluetooth / 3.5mm jack"),
        ("Fun Fact",   "Sounds above 85dB can damage hearing"),
    ],
    "geometry box": [
        ("Category",   "Stationery / Math Tool"),
        ("Used For",   "Drawing angles, circles, lines"),
        ("Contains",   "Compass, Protractor, Ruler, Set squares"),
        ("Used By",    "Students"),
        ("Fun Fact",   "Compass was invented ~2500 years ago"),
    ],
    "mobile,phone": [
        ("Category",   "Electronics / Communication"),
        ("Used For",   "Calls, messages, internet, camera"),
        ("Material",   "Glass, Aluminium, Plastic"),
        ("OS",         "Android or iOS"),
        ("Fun Fact",   "6.9 billion smartphones exist worldwide"),
    ],
    "notebook": [
        ("Category",   "Stationery / Writing"),
        ("Used For",   "Notes, assignments, diary"),
        ("Material",   "Paper with Cardboard cover"),
        ("Page Types", "Ruled, Plain, Grid"),
        ("Fun Fact",   "The word 'notebook' dates back to 1579"),
    ],
    "pen": [
        ("Category",   "Stationery / Writing"),
        ("Used For",   "Writing and drawing"),
        ("Ink Type",   "Ballpoint / Gel / Fountain"),
        ("Invented",   "Ballpoint by Lazlo Biro, 1938"),
        ("Fun Fact",   "One pen can write ~45,000 words"),
    ],
    "scale": [
        ("Category",   "Stationery / Measuring"),
        ("Used For",   "Measuring length, drawing lines"),
        ("Material",   "Plastic or Metal"),
        ("Measures",   "cm and inches"),
        ("Fun Fact",   "Also called a Ruler"),
    ],
    "stapler": [
        ("Category",   "Office / Stationery"),
        ("Used For",   "Binding papers together"),
        ("Material",   "Metal & Plastic"),
        ("Invented",   "1866 by Henry R. Heyl"),
        ("Fun Fact",   "Holds ~200 staple pins"),
    ],
    "water bottle": [
        ("Category",   "Utility / Drinkware"),
        ("Used For",   "Storing and drinking water"),
        ("Material",   "Plastic, Steel, or Glass"),
        ("Capacity",   "500 ml to 1 Litre"),
        ("Fun Fact",   "Humans need ~2-3 litres of water daily"),
    ],
    "whitener": [
        ("Category",   "Stationery / Correction"),
        ("Used For",   "Covering writing mistakes"),
        ("Dries In",   "30-60 seconds"),
        ("Also Known", "Correction fluid / Tipp-Ex"),
        ("Fun Fact",   "Invented by Bette Nesmith Graham, 1956"),
    ],
}
# ──────────────────────────────────────────────────────────────────────────────

# Colors (BGR)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_GREEN  = (0,   210, 90)
C_YELLOW = (30,  220, 220)
C_RED    = (60,  60,  240)
C_CYAN   = (230, 200, 50)
C_GRAY   = (120, 120, 120)
C_PANEL  = (25,  25,  25)
C_SCAN   = (0,   170, 255)
C_HEADER = (50,  30,  10)


def load_models():
    """Load all saved .pkl models from disk."""
    models = []
    for full_name, pkl_file in MODEL_FILES:
        path = os.path.join(MODELS_DIR, pkl_file)
        if os.path.exists(path):
            models.append((full_name, joblib.load(path)))
            print(f"  Loaded : {full_name}")
        else:
            print(f"  [WARN]  Not found: {path}")
    return models   # list of (name, model)


def preprocess_frame(frame):
    """Extract features from webcam frame — same pipeline as training."""
    return extract_features(frame, augment=False).reshape(1, -1)


def run_all_models(models, features):
    """
    Run every model on the features.
    Returns:
      algo_results  : list of (full_name, predicted_label, confidence%)
      majority      : label with most votes
      votes         : dict {label: vote_count}
    """
    algo_results = []
    votes        = {}

    for full_name, model in models:
        pred_id    = model.predict(features)[0]
        pred_label = CLASS_NAMES[pred_id]

        if hasattr(model, "predict_proba"):
            conf = model.predict_proba(features)[0][pred_id] * 100
        else:
            conf = 0.0

        algo_results.append((full_name, pred_label, conf))
        votes[pred_label] = votes.get(pred_label, 0) + 1

    majority = max(votes, key=votes.get)
    return algo_results, majority, votes


def draw_text(canvas, text, x, y, scale, color, thickness=1):
    cv2.putText(canvas, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _get_object_box(frame):
    """Return (x1,y1,x2,y2) bounding box of detected object, or None."""
    from data_preparation import _find_object_crop as _foc
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
        if cv2.contourArea(largest) > h * w * 0.03:
            x, y, bw, bh = cv2.boundingRect(largest)
            px, py = int(bw*0.08), int(bh*0.08)
            return (max(0,x-px), max(0,y-py), min(w,x+bw+px), min(h,y+bh+py))
    return None


def draw_scanning_overlay(frame):
    """Live scanning mode — show auto-detected object box in real time."""
    canvas = frame.copy()
    h, w   = canvas.shape[:2]

    # Try to detect object and draw box around it
    box = _get_object_box(frame)
    if box:
        x1, y1, x2, y2 = box
        # Dim outside detected box
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        canvas[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        # Green box = object detected
        cv2.rectangle(canvas, (x1,y1), (x2,y2), C_GREEN, 2)
        c = 20
        for (bx,by,sx,sy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(canvas, (bx,by), (bx+sx*c, by), C_GREEN, 3)
            cv2.line(canvas, (bx,by), (bx, by+sy*c), C_GREEN, 3)
        draw_text(canvas, "Object detected - SPACE to Identify", x1, max(y1-8,16), 0.45, C_GREEN)
    else:
        # No object found — guide user
        draw_text(canvas, "No object detected. Hold object closer to camera.", 10, 30, 0.5, C_SCAN)

    # Bottom instruction bar
    cv2.rectangle(canvas, (0, h-38), (w, h), C_BLACK, -1)
    draw_text(canvas, "SPACE = Identify   R = Rescan   Q = Quit", 10, h-12, 0.5, C_SCAN)

    return canvas


def draw_result_screen(frozen_frame, algo_results, majority, votes):
    """
    Full result screen:
      LEFT  : frozen webcam frame with object name stamped
      RIGHT : panel with all 6 algorithm predictions + object properties
    """
    h, w    = frozen_frame.shape[:2]
    panel_w = 420
    canvas  = np.zeros((h, w + panel_w, 3), dtype=np.uint8)

    # ── Left: frozen frame (slightly dimmed) ──────────────────────────────────
    canvas[:h, :w] = frozen_frame
    dim = np.zeros_like(frozen_frame)
    cv2.addWeighted(frozen_frame, 0.55, dim, 0.45, 0, canvas[:h, :w])

    # Object name stamp on frozen frame
    obj_upper = majority.upper()
    fs = 1.1
    (tw, th), _ = cv2.getTextSize(obj_upper, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
    tx = (w - tw) // 2
    ty = h // 2 + th // 2
    # Shadow
    draw_text(canvas, obj_upper, tx + 2, ty + 2, fs, C_BLACK, 3)
    draw_text(canvas, obj_upper, tx,     ty,     fs, C_GREEN,  3)

    agree = votes.get(majority, 0)
    sub   = f"{agree}/6 algorithms agree"
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    draw_text(canvas, sub, (w - sw) // 2, ty + 30, 0.52, C_YELLOW)

    # ── Right panel ───────────────────────────────────────────────────────────
    canvas[:, w:] = C_PANEL
    F   = cv2.FONT_HERSHEY_SIMPLEX
    px  = w + 10
    pw  = panel_w - 18   # usable width

    # Header
    cv2.rectangle(canvas, (w, 0), (w + panel_w, 42), (10, 10, 50), -1)
    draw_text(canvas, "OBJECT IDENTIFIED", px, 18, 0.58, C_YELLOW, 2)
    draw_text(canvas, f"Final Answer: {majority}", px, 36, 0.4, C_WHITE)

    y = 50

    # ── Section 1: All 6 Algorithm Results ────────────────────────────────────
    cv2.line(canvas, (w+4, y), (w+panel_w-4, y), C_CYAN, 1)
    y += 14
    draw_text(canvas, "ALL 6 ALGORITHM PREDICTIONS:", px, y, 0.42, C_CYAN, 1)
    y += 16

    for full_name, pred_label, conf in algo_results:
        is_winner = (pred_label == majority)
        row_col   = C_GREEN if is_winner else C_GRAY

        # Algorithm name
        draw_text(canvas, full_name + ":", px, y, 0.37, C_YELLOW)
        y += 14

        # Prediction label + confidence
        pred_txt = f"  -> {pred_label}  ({conf:.1f}%)"
        draw_text(canvas, pred_txt, px, y, 0.38, row_col)

        # Confidence bar
        bar_max = pw - 10
        bar_len = int((conf / 100.0) * bar_max)
        by = y + 3
        cv2.rectangle(canvas, (px, by), (px + bar_max, by + 7), (50,50,50), -1)
        cv2.rectangle(canvas, (px, by), (px + bar_len,  by + 7), row_col,   -1)
        y += 20

    # ── Section 2: Object Properties ──────────────────────────────────────────
    cv2.line(canvas, (w+4, y), (w+panel_w-4, y), C_CYAN, 1)
    y += 14
    draw_text(canvas, "OBJECT PROPERTIES:", px, y, 0.42, C_CYAN, 1)
    y += 16

    props = OBJECT_PROPERTIES.get(majority, [])
    for key, value in props:
        if y > h - 50:
            break
        draw_text(canvas, f"{key}:", px, y, 0.37, C_YELLOW)
        y += 14
        # Wrap value text if too long
        words, line = value.split(), ""
        for word in words:
            test = (line + " " + word).strip()
            (tw2, _), _ = cv2.getTextSize("  " + test, F, 0.37, 1)
            if tw2 > pw - 10 and line:
                draw_text(canvas, "  " + line, px, y, 0.37, C_WHITE)
                y    += 14
                line  = word
            else:
                line = test
        if line:
            draw_text(canvas, "  " + line, px, y, 0.37, C_WHITE)
            y += 18

    # ── Footer ────────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (w, h-30), (w+panel_w, h), (10,10,10), -1)
    draw_text(canvas, "SPACE / R = Scan Again    Q = Quit", px, h-10, 0.38, C_RED)

    return canvas


def run_webcam():
    print("\n" + "=" * 58)
    print("  SMART OBJECT IDENTIFIER  —  6 Algorithm Comparison")
    print("=" * 58)
    print("  Loading models ...")

    models = load_models()
    if not models:
        print("\n[ERROR] No models found. Run main.py first.")
        return

    print(f"\n  {len(models)}/6 models loaded.")
    print("\n  Controls:")
    print("    SPACE  ->  Freeze frame & identify object")
    print("    R      ->  Back to live scanning")
    print("    Q/ESC  ->  Quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state        = "scanning"    # "scanning" | "identified"
    frozen_frame = None
    algo_results = []
    majority     = ""
    votes        = {}

    WIN = "Smart Object Identifier  |  SPACE=Identify  R=Rescan  Q=Quit"

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if state == "scanning":
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            cv2.imshow(WIN, draw_scanning_overlay(frame))

            if key == ord(" "):
                features                    = preprocess_frame(frame)
                algo_results, majority, votes = run_all_models(models, features)
                frozen_frame                = frame.copy()
                state                       = "identified"
                print(f"\n  Identified: {majority}  ({votes.get(majority,0)}/6 agree)")
                for n, l, c in algo_results:
                    print(f"    {n:<30}  {l:<15}  {c:.1f}%")

        elif state == "identified":
            cv2.imshow(WIN, draw_result_screen(frozen_frame, algo_results, majority, votes))

            if key in (ord(" "), ord("r"), ord("R")):
                state = "scanning"

    cap.release()
    cv2.destroyAllWindows()
    print("\nGoodbye!")


if __name__ == "__main__":
    run_webcam()
