# Gesture Playground

Hand gesture recognition with a webcam — desktop (Python) and browser (MediaPipe).

## Gestures

Fist, Thumbs Up, Point, Peace, Three, Four, Open Palm, Gun, OK, Call Me, Rock, Love You, Pinky

## Desktop app

### Requirements

- Python 3.10+
- Webcam

```bash
pip install opencv-python mediapipe numpy
python hand_gesture.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` / Esc | Quit |
| `h` | Toggle help |
| `d` | Toggle landmarks |
| `m` | Toggle mirror |
| `s` | Screenshot |

## Web demo

Open `web/index.html` via a local server (camera requires `localhost` or HTTPS):

```bash
cd web
python -m http.server 8765
```

Then visit http://127.0.0.1:8765/

Click **Start camera**, allow permission, and try the gestures.
