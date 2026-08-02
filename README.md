# Gesture Playground

Live webcam hand-gesture recognition using [MediaPipe](https://developers.google.com/mediapipe).  
Works in the browser and as a local Python app.

## Try it live

**[https://hand-gestures-playground.netlify.app/](https://hand-gestures-playground.netlify.app/)**

1. Open the link  
2. Click **Start camera** and allow access  
3. Show your hand to the camera and try the gestures below  

Camera video stays on your device — nothing is uploaded.

---

## Supported gestures

| Gesture | How to make it |
|---------|----------------|
| Fist | All fingers down |
| Open Palm | All five fingers up |
| Point | Index only |
| Peace | Index + middle |
| Three | Index + middle + ring |
| Four | Four fingers (no thumb) |
| Thumbs Up | Thumb only |
| Gun | Thumb + index |
| OK | Thumb tip touches index tip |
| Call Me | Thumb + pinky |
| Rock | Index + pinky |
| Love You | Thumb + index + pinky |
| Pinky | Pinky only |

Tips: good lighting, palm facing the camera, and keep your hand clearly in frame.

---

## Project layout

```
hand-gestures/
├── hand_gesture.py      # Desktop OpenCV app
├── requirements.txt
├── web/
│   └── index.html       # Browser demo (also deployed on Netlify)
└── README.md
```

---

## Run locally (browser)

```bash
cd web
python -m http.server 8765
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/), then start the camera.

> Camera access needs `localhost` or HTTPS (the Netlify link already uses HTTPS).

---

## Run locally (desktop)

### Requirements

- Python 3.10+
- A webcam

```bash
pip install -r requirements.txt
python hand_gesture.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` / Esc | Quit |
| `h` | Toggle help |
| `d` | Toggle landmarks |
| `m` | Toggle mirror |
| `s` | Save screenshot |

Optional flags:

```bash
python hand_gesture.py --camera 0 --hands 2
python hand_gesture.py --no-mirror
```

---

## How it works

- **Hands:** MediaPipe Hand Landmarker tracks up to two hands and Left/Right labels  
- **Fingers:** Index–pinky use tip vs PIP height; thumb uses a hand-local frame (pinky→index / wrist→middle) so open vs closed works for both hands  
- **Gestures:** Finger patterns are mapped to named poses, with light temporal smoothing to reduce flicker  

---

## License

Use and share freely for learning and demos.
