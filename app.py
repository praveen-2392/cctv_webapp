from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
import cv2
import numpy as np
import time
import os
import threading
import datetime
import requests as req
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key    = "cctv_secret_2025"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload limit

# ================= LOGIN =================
USERS = {
    "admin": "admin123",
    "user":  "user123"
}

# ================= TELEGRAM =================
TOKEN   = "8774920390:AAEgj8qpE-FgMDpMDVMP1vXa1_emn4QUmvg"
CHAT_ID = "8363809372"

def tg_send(endpoint, data=None, files=None):
    try:
        req.post(
            f"https://api.telegram.org/bot{TOKEN}/{endpoint}",
            data=data, files=files, timeout=15
        )
    except: pass

def send_message(text):
    threading.Thread(
        target=tg_send, args=("sendMessage",),
        kwargs={"data": {"chat_id": CHAT_ID, "text": text}},
        daemon=True
    ).start()

def send_image(path):
    def _t():
        try:
            with open(path, "rb") as f:
                tg_send("sendPhoto", data={"chat_id": CHAT_ID}, files={"photo": f})
        except: pass
    threading.Thread(target=_t, daemon=True).start()

def send_video_file(path):
    def _t():
        try:
            with open(path, "rb") as f:
                tg_send("sendVideo", data={"chat_id": CHAT_ID}, files={"video": f})
        except: pass
    threading.Thread(target=_t, daemon=True).start()

# ================= MODEL FIX =================
class FixedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs:
            kwargs["shape"] = tuple(kwargs.pop("batch_shape")[1:])
        super().__init__(*args, **kwargs)

# ================= GLOBAL STATE =================
state = {
    "running":        False,
    "paused":         False,
    "mode":           "live",      # "live" or "video"
    "source":         0,
    "video_path":     None,
    "video_name":     "",
    "frame":          None,
    "status":         "OFFLINE",
    "score":          0.0,
    "violence_th":    0.35,
    "persons":        0,
    "alerts":         [],
    "total_alerts":   0,
    "violence_count": 0,
    "weapon_count":   0,
    "start_time":     None,
    "runtime":        "00:00:00",
    # video progress
    "video_frame":    0,
    "video_total":    0,
    "video_time":     "00:00",
    "video_duration": "00:00",
    "lock":           threading.Lock()
}

models_loaded  = False
person_model   = None
weapon_model   = None
violence_model = None

ALLOWED = {"mp4", "avi", "mov", "mkv", "wmv", "flv"}

os.makedirs("snapshots",           exist_ok=True)
os.makedirs("evidence_clips",      exist_ok=True)
os.makedirs("static/snapshots",    exist_ok=True)
os.makedirs("uploads",             exist_ok=True)

# ================= LOAD MODELS =================
def load_models():
    global person_model, weapon_model, violence_model, models_loaded
    try:
        print("Loading models...")
        person_model   = YOLO("yolov8s.pt")
        weapon_model   = YOLO("best.pt")
        violence_model = load_model(
            "violence_detection_model.h5",
            compile=False,
            custom_objects={"InputLayer": FixedInputLayer}
        )
        models_loaded = True
        print("All models ready")
    except Exception as e:
        print(f"Model load error: {e}")
        models_loaded = False

threading.Thread(target=load_models, daemon=True).start()

# ================= HELPERS =================
WEAPON_COLORS = {"gun": (0,0,255), "knife": (0,60,255)}

def enhance(frame):
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def ts_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def now_str():
    return datetime.datetime.now().strftime("%H:%M:%S")

def fmt_sec(s):
    s = int(s)
    return f"{s//60:02d}:{s%60:02d}"

def add_alert(atype, detail):
    with state["lock"]:
        state["alerts"].insert(0, {
            "time":   now_str(),
            "type":   atype,
            "detail": detail
        })
        state["alerts"]         = state["alerts"][:60]
        state["total_alerts"]  += 1
        if atype in ("VIOLENCE", "HIGH ALERT"):
            state["violence_count"] += 1
        elif atype == "WEAPON":
            state["weapon_count"]   += 1

# ================= DETECTION CORE =================
def run_detection(cap, is_video=False):
    """
    Shared detection loop used by both live camera and video file modes.
    cap      — OpenCV VideoCapture already opened
    is_video — True when running on uploaded video file
    """
    frame_buffer   = []
    pred_buffer    = deque(maxlen=6)
    violence_votes = 0
    last_v_alert   = -999
    last_w_alert   = {}
    recording      = False
    video_writer   = None
    record_start   = 0
    vid_path       = None
    prev_gray      = None
    frame_count    = 0
    fail_count     = 0
    cached_persons = []
    cached_weapons = []

    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total_fr    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0
    frame_delay = 1.0 / video_fps if is_video else 0

    while state["running"]:

        if state["paused"]:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()

        if not ret:
            if is_video:
                # Video ended
                break
            else:
                fail_count += 1
                if fail_count > 30:
                    cap.release()
                    time.sleep(2)
                    cap        = cv2.VideoCapture(state["source"])
                    fail_count = 0
                continue

        fail_count   = 0
        frame_count += 1
        now_real     = time.time()
        th           = state["violence_th"]

        # Update video progress
        if is_video and total_fr > 0:
            with state["lock"]:
                state["video_frame"]    = frame_count
                state["video_total"]    = total_fr
                state["video_time"]     = fmt_sec(frame_count / video_fps)
                state["video_duration"] = fmt_sec(total_fr / video_fps)

        frame   = cv2.resize(frame, (640, 480))
        frame   = enhance(frame)
        display = frame.copy()

        # YOLO every 3rd frame
        if frame_count % 3 == 0 and models_loaded:
            p_res        = person_model(frame, conf=0.20, verbose=False)[0]
            person_boxes = [tuple(map(int, b.xyxy[0]))
                            for b in p_res.boxes if int(b.cls[0]) == 0]
            w_res        = weapon_model(frame, conf=0.25, verbose=False)[0]
            weps = []
            for b in w_res.boxes:
                cid          = int(b.cls[0])
                cf           = float(b.conf[0])
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                nm           = w_res.names[cid].lower()
                weps.append((nm, cf, x1, y1, x2, y2))
            cached_persons = person_boxes
            cached_weapons = weps
        else:
            person_boxes = cached_persons
            weps         = cached_weapons

        # Draw persons
        for (x1,y1,x2,y2) in person_boxes:
            cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(display,"Person",(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # Draw weapons + alert
        for (nm, cf, x1, y1, x2, y2) in weps:
            col = WEAPON_COLORS.get(nm,(0,0,255))
            cv2.rectangle(display,(x1,y1),(x2,y2),col,3)
            lbl        = f"{nm.upper()} {cf:.2f}"
            (tw,th_),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            cv2.rectangle(display,(x1,y1-th_-12),(x1+tw+8,y1),col,-1)
            cv2.putText(display,lbl,(x1+4,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            if now_real - last_w_alert.get(nm,-999) > 20:
                snap = f"static/snapshots/weapon_{ts_str()}.jpg"
                cv2.imwrite(snap, display)
                vt   = f" | Video: {fmt_sec(frame_count/video_fps)}" if is_video else ""
                msg  = (f"WEAPON DETECTED\n"
                        f"Weapon: {nm.upper()}\nConf: {cf:.0%}\n"
                        f"Time: {now_str()}{vt}")
                send_message(msg)
                send_image(snap)
                last_w_alert[nm] = now_real
                add_alert("WEAPON", f"{nm.upper()} {cf:.0%}")

        # Motion
        motion = False
        gray   = cv2.GaussianBlur(
            cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(21,21),0)
        if prev_gray is not None:
            diff   = cv2.absdiff(prev_gray, gray)
            thr    = cv2.threshold(diff,10,255,cv2.THRESH_BINARY)[1]
            for (x1,y1,x2,y2) in person_boxes:
                roi = thr[y1:y2, x1:x2]
                if roi.size > 0 and np.sum(roi)/255/roi.size > 0.02:
                    motion = True; break
        prev_gray = gray

        # Violence
        score   = 0.0
        v_label = "NO PERSON"
        v_color = (120,120,120)

        if frame_count % 3 == 0 and models_loaded:
            if person_boxes and motion:
                x1,y1,x2,y2 = max(person_boxes,
                                   key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    frame_buffer.append(
                        cv2.resize(crop,(64,64)).astype(np.float32)/255.0)
                if len(frame_buffer) > 6: frame_buffer.pop(0)
                if len(frame_buffer) == 6:
                    score = float(violence_model.predict(
                        np.expand_dims(frame_buffer,axis=0),verbose=0)[0][0])
                    pred_buffer.append(score)
                    avg   = float(np.mean(pred_buffer))
                    score = avg
                    if avg > th: violence_votes += 1
                    else:        violence_votes = max(0, violence_votes-1)
                    if violence_votes >= 3:
                        v_label = f"VIOLENCE {avg:.2f}"
                        v_color = (0,0,255)
                        if now_real - last_v_alert > 30:
                            snap  = f"static/snapshots/violence_{ts_str()}.jpg"
                            vpath = f"evidence_clips/violence_{ts_str()}.avi"
                            cv2.imwrite(snap, display)
                            w_nm  = [w[0].upper() for w in weps]
                            vt    = f" | Video: {fmt_sec(frame_count/video_fps)}" if is_video else ""
                            if w_nm:
                                msg = (f"HIGH ALERT: VIOLENCE + {','.join(w_nm)}\n"
                                       f"Score: {avg:.0%}\nTime: {now_str()}{vt}")
                                add_alert("HIGH ALERT", f"Violence+{','.join(w_nm)} {avg:.0%}")
                            else:
                                msg = (f"VIOLENCE DETECTED\n"
                                       f"Score: {avg:.0%}\nTime: {now_str()}{vt}")
                                add_alert("VIOLENCE", f"Score {avg:.0%}")
                            vw = cv2.VideoWriter(
                                vpath,cv2.VideoWriter_fourcc(*'XVID'),15,(640,480))
                            recording    = True
                            record_start = now_real
                            vid_path     = vpath
                            send_message(msg)
                            send_image(snap)
                            last_v_alert   = now_real
                            violence_votes = 0
                    else:
                        v_label = f"NORMAL {avg:.2f}"
                        v_color = (0,200,0)
            elif person_boxes:
                v_label = "PERSON STILL"
                v_color = (200,200,0)
                frame_buffer.clear(); pred_buffer.clear(); violence_votes = 0
            else:
                frame_buffer.clear(); pred_buffer.clear(); violence_votes = 0

        # Recording evidence clip
        if recording and video_writer is not None:
            video_writer.write(display)
            if now_real - record_start >= 10:
                recording = False
                video_writer.release()
                video_writer = None
                send_video_file(vid_path)

        # Overlay on frame
        cv2.rectangle(display,(0,0),(640,55),(0,0,0),-1)
        lbl = v_label
        col = v_color
        if weps:
            wn  = "+".join(set(w[0].upper() for w in weps))
            lbl = f"WARNING: {wn}"
            col = (0,0,255)
        cv2.putText(display,lbl,(10,32),cv2.FONT_HERSHEY_SIMPLEX,0.85,col,2)
        bw = int(score*200)
        bc = (0,255,0) if score < th else (0,0,255)
        cv2.rectangle(display,(10,40),(210,52),(40,40,40),-1)
        cv2.rectangle(display,(10,40),(10+bw,52),bc,-1)
        cv2.putText(display,
                    f"score:{score:.2f}  th:{th:.2f}  persons:{len(person_boxes)}",
                    (215,50),cv2.FONT_HERSHEY_SIMPLEX,0.36,(160,160,160),1)

        # Video mode — show progress on frame
        if is_video and total_fr > 0:
            prog = int((frame_count / total_fr) * 620)
            cv2.rectangle(display,(0,470),(640,480),(20,20,20),-1)
            cv2.rectangle(display,(0,470),(prog,480),(0,180,80),-1)
            cv2.putText(display,
                        f"{fmt_sec(frame_count/video_fps)} / {fmt_sec(total_fr/video_fps)}",
                        (8,478),cv2.FONT_HERSHEY_SIMPLEX,0.38,(160,160,160),1)
        else:
            cv2.rectangle(display,(0,458),(640,480),(0,0,0),-1)
            cv2.putText(display,
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        (8,474),cv2.FONT_HERSHEY_SIMPLEX,0.42,(160,160,160),1)

        if recording:
            cv2.circle(display,(615,20),8,(0,0,255),-1)

        # Update state
        with state["lock"]:
            state["frame"]   = display.copy()
            state["status"]  = lbl
            state["score"]   = round(score, 3)
            state["persons"] = len(person_boxes)
            if state["start_time"]:
                s = int(now_real - state["start_time"])
                state["runtime"] = f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

        # Maintain correct playback speed for video
        if is_video:
            time.sleep(max(0, frame_delay - 0.01))

    cap.release()
    with state["lock"]:
        state["running"] = False
        state["status"]  = "ANALYSIS COMPLETE" if is_video else "OFFLINE"
        state["frame"]   = None

# ================= THREAD LAUNCHERS =================
def start_live_thread(source):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,     1)
    for _ in range(5): cap.read()
    run_detection(cap, is_video=False)

def start_video_thread(path):
    cap = cv2.VideoCapture(path)
    run_detection(cap, is_video=True)

# ================= STREAM =================
def gen_frames():
    while True:
        with state["lock"]:
            frame = state.get("frame")
        if frame is None:
            blank = np.zeros((480,640,3), np.uint8)
            mode  = state["mode"]
            if mode == "video" and not state["running"]:
                cv2.putText(blank,"Upload a video to analyse",
                            (120,240),cv2.FONT_HERSHEY_SIMPLEX,0.8,(60,60,60),2)
            else:
                cv2.putText(blank,"Camera Offline",
                            (200,240),cv2.FONT_HERSHEY_SIMPLEX,1,(60,60,60),2)
            frame = blank
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY,70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.04)

# ================= ROUTES =================
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))

@app.route("/login", methods=["GET","POST"])
def login():
    error = ""
    if request.method == "POST":
        u = request.form.get("username","")
        p = request.form.get("password","")
        if USERS.get(u) == p:
            session["user"] = u
            return redirect(url_for("dashboard"))
        error = "Invalid username or password"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

@app.route("/video_feed")
def video_feed():
    if "user" not in session: return "", 403
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ---- Live camera ----
@app.route("/api/start_live", methods=["POST"])
def start_live():
    if "user" not in session: return jsonify({"ok": False})
    if state["running"]:      return jsonify({"ok": False, "msg": "Already running"})
    data   = request.json or {}
    source = data.get("source", 0)
    try:    source = int(source)
    except: pass
    state.update({
        "running": True, "paused": False, "mode": "live",
        "source": source, "start_time": time.time(),
        "alerts": [], "total_alerts": 0,
        "violence_count": 0, "weapon_count": 0
    })
    threading.Thread(target=start_live_thread, args=(source,), daemon=True).start()
    send_message(f"CCTV Live Started\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return jsonify({"ok": True})

# ---- Video upload ----
@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "user" not in session: return jsonify({"ok": False, "msg": "Not logged in"})
    if state["running"]:      return jsonify({"ok": False, "msg": "Stop current session first"})
    if "video" not in request.files:
        return jsonify({"ok": False, "msg": "No file uploaded"})
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"ok": False, "msg": "Empty filename"})
    ext = f.filename.rsplit(".",1)[-1].lower()
    if ext not in ALLOWED:
        return jsonify({"ok": False, "msg": f"File type .{ext} not allowed"})
    filename = secure_filename(f.filename)
    save_path = os.path.join("uploads", filename)
    f.save(save_path)
    state.update({
        "running": True, "paused": False, "mode": "video",
        "video_path": save_path, "video_name": filename,
        "start_time": time.time(),
        "alerts": [], "total_alerts": 0,
        "violence_count": 0, "weapon_count": 0,
        "video_frame": 0, "video_total": 0
    })
    threading.Thread(
        target=start_video_thread, args=(save_path,), daemon=True
    ).start()
    send_message(f"Video Analysis Started\nFile: {filename}\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return jsonify({"ok": True, "filename": filename})

# ---- Stop ----
@app.route("/api/stop", methods=["POST"])
def stop():
    if "user" not in session: return jsonify({"ok": False})
    state["running"] = False
    send_message(
        f"Session Stopped\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total alerts: {state['total_alerts']}"
    )
    return jsonify({"ok": True})

# ---- Pause ----
@app.route("/api/pause", methods=["POST"])
def pause():
    if "user" not in session: return jsonify({"ok": False})
    state["paused"] = not state["paused"]
    return jsonify({"paused": state["paused"]})

# ---- Threshold ----
@app.route("/api/set_threshold", methods=["POST"])
def set_threshold():
    if "user" not in session: return jsonify({"ok": False})
    state["violence_th"] = float(request.json.get("value", 0.35))
    return jsonify({"ok": True})

# ---- Status ----
@app.route("/api/status")
def get_status():
    if "user" not in session: return jsonify({})
    with state["lock"]:
        return jsonify({
            "running":        state["running"],
            "paused":         state["paused"],
            "mode":           state["mode"],
            "status":         state["status"],
            "score":          state["score"],
            "persons":        state["persons"],
            "total_alerts":   state["total_alerts"],
            "violence_count": state["violence_count"],
            "weapon_count":   state["weapon_count"],
            "runtime":        state.get("runtime","00:00:00"),
            "video_name":     state.get("video_name",""),
            "video_time":     state.get("video_time","00:00"),
            "video_duration": state.get("video_duration","00:00"),
            "video_frame":    state.get("video_frame",0),
            "video_total":    state.get("video_total",0),
            "alerts":         state["alerts"][:12],
            "models_loaded":  models_loaded
        })

# ---- Manual snapshot ----
@app.route("/api/snapshot", methods=["POST"])
def snapshot():
    if "user" not in session: return jsonify({"ok": False})
    with state["lock"]:
        frame = state.get("frame")
    if frame is not None:
        path = f"static/snapshots/manual_{ts_str()}.jpg"
        cv2.imwrite(path, frame)
        return jsonify({"ok": True})
    return jsonify({"ok": False})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)