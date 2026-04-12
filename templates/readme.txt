AI CCTV Web App — Setup Guide
==============================

FOLDER STRUCTURE
----------------
cctv_webapp/
├── app.py
├── requirements.txt
├── best.pt                        ← copy your weapon model here
├── violence_detection_model.h5    ← copy your violence model here
├── yolov8s.pt                     ← auto downloads on first run
├── templates/
│   ├── login.html
│   └── dashboard.html
└── static/
    └── snapshots/                 ← auto created


SETUP STEPS
-----------
1. Copy these files into your project folder:
   - best.pt
   - violence_detection_model.h5

2. Install requirements:
   pip install -r requirements.txt

3. Run the app:
   python app.py

4. Open browser:
   http://localhost:5000

5. Login:
   Username: admin
   Password: admin123


DASHBOARD FEATURES
------------------
- Live camera stream in browser
- START / STOP / PAUSE controls
- Violence threshold slider (adjust in real time)
- Live score bar showing model output
- Alert log with timestamps
- Stats: runtime, total alerts, violence count, weapon count
- Manual snapshot button
- Telegram alerts sent automatically


CAMERA SETUP
------------
Webcam:      source = 0
IP Camera:   source = rtsp://admin:password@192.168.1.64/stream

Change the CAMERA source by typing in the input box on the dashboard.


LOGIN CREDENTIALS
-----------------
admin / admin123
user  / user123

To change passwords, edit USERS dict in app.py:
USERS = {
    "admin": "your_new_password",
    "user":  "another_password"
