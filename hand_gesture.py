"""
Hand gesture detector using MediaPipe Hands.

MediaPipe handedness assumes a mirrored (selfie) image. This app flips the
frame when mirror mode is on, then trusts MediaPipe's Left/Right labels.

Thumb open/closed is decided in a hand-local coordinate frame built from
wrist / index-MCP / pinky-MCP — not from distance to other fingertips
(those distances caused false open/closed results).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17
WRIST = 0

FINGER_PAIRS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)


def _dist(a, b) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)


def _xy(lm) -> np.ndarray:
    return np.array([lm.x, lm.y], dtype=np.float64)


@dataclass(frozen=True)
class HandReading:
    hand_side: str
    confidence: float
    fingers: Tuple[bool, bool, bool, bool, bool]
    gesture: str
    bbox: Tuple[int, int, int, int]


class HandGestureDetector:
    GESTURES = {
        (False, False, False, False, False): "Fist",
        (True, False, False, False, False): "Thumbs Up",
        (False, True, False, False, False): "Point",
        (False, True, True, False, False): "Peace",
        (False, True, True, True, False): "Three",
        (False, True, True, True, True): "Four",
        (True, True, True, True, True): "Open Palm",
        (True, True, False, False, False): "Gun",
        (True, False, False, False, True): "Call Me",
        (False, True, False, False, True): "Rock",
        (True, True, False, False, True): "Love You",
        (False, False, False, False, True): "Pinky",
    }

    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        mirror: bool = True,
        history_size: int = 7,
    ) -> None:
        self.mirror = mirror
        self.history_size = max(1, history_size)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self._histories: dict[str, Deque[str]] = {}

    def close(self) -> None:
        self.hands.close()

    def display_hand_side(self, mediapipe_label: str) -> str:
        """
        MediaPipe assumes mirrored input.
        - Mirror on  (we flip first): trust the label.
        - Mirror off (raw camera): invert the label for the person.
        """
        if self.mirror:
            return mediapipe_label
        return "Right" if mediapipe_label == "Left" else "Left"

    @staticmethod
    def is_thumb_extended(landmarks: Sequence) -> bool:
        """
        Decide thumb open/closed in a hand-local 2D frame.

        Axes (anatomical, same for left/right and palm/back):
          x = pinky MCP -> index MCP  (toward the thumb side of the palm)
          y = wrist -> middle MCP     (toward the fingertips)

        Open:
          - Abducted (open palm / gun): tip further along +x than IP
          - Thumbs up: tip further along -y than IP

        Closed (fist / tucked / beside fingers):
          - tip stays near MCP or curls toward pinky / into the fingers
        """
        wrist = _xy(landmarks[WRIST])
        index_mcp = _xy(landmarks[INDEX_MCP])
        middle_mcp = _xy(landmarks[MIDDLE_MCP])
        pinky_mcp = _xy(landmarks[PINKY_MCP])
        tip = _xy(landmarks[THUMB_TIP])
        ip = _xy(landmarks[THUMB_IP])
        mcp = _xy(landmarks[THUMB_MCP])

        y_axis = middle_mcp - wrist
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm < 1e-6:
            return False
        y_axis /= y_norm

        x_axis = index_mcp - pinky_mcp
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-6:
            return False
        x_axis /= x_norm

        hand_size = y_norm

        def local(pt: np.ndarray) -> Tuple[float, float]:
            v = pt - mcp
            return (
                float(np.dot(v, x_axis) / hand_size),
                float(np.dot(v, y_axis) / hand_size),
            )

        tip_x, tip_y = local(tip)
        ip_x, ip_y = local(ip)

        abducted = (tip_x > ip_x + 0.10) and (tip_x > 0.12)
        thumbs_up = (tip_y < ip_y - 0.10) and (tip_y < -0.08)
        return abducted or thumbs_up

    def get_finger_states(
        self, landmarks: Sequence, mp_label: str | None = None
    ) -> Tuple[bool, bool, bool, bool, bool]:
        del mp_label  # thumb no longer depends on Left/Right label
        thumb_up = self.is_thumb_extended(landmarks)
        others = tuple(
            landmarks[tip].y < landmarks[pip].y - 0.01 for tip, pip in FINGER_PAIRS
        )
        return (thumb_up, *others)

    def recognize_gesture(
        self,
        fingers: Sequence[bool],
        landmarks: Sequence | None = None,
    ) -> str:
        if landmarks is not None:
            tip_t, tip_i = landmarks[THUMB_TIP], landmarks[INDEX_TIP]
            # OK naming only — does not drive thumb open/closed state.
            if (
                _dist(tip_t, tip_i) < 0.05
                and not fingers[2]
                and not fingers[3]
                and not fingers[4]
            ):
                return "OK"

        key = tuple(bool(f) for f in fingers)
        if key in self.GESTURES:
            return self.GESTURES[key]

        count = sum(key)
        if count == 0:
            return "Fist"
        if count == 5:
            return "Open Palm"
        return f"Custom ({count} up)"

    def smooth_gesture(self, hand_key: str, gesture: str) -> str:
        history = self._histories.setdefault(
            hand_key, deque(maxlen=self.history_size)
        )
        history.append(gesture)
        counts = Counter(history)
        best_count = max(counts.values())
        for g in reversed(history):
            if counts[g] == best_count:
                return g
        return gesture

    def hand_bbox(
        self, landmarks: Sequence, width: int, height: int, pad: int = 12
    ) -> Tuple[int, int, int, int]:
        xs = [int(lm.x * width) for lm in landmarks]
        ys = [int(lm.y * height) for lm in landmarks]
        return (
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            min(width - 1, max(xs) + pad),
            min(height - 1, max(ys) + pad),
        )

    def prune_histories(self, active_keys: set[str]) -> None:
        for key in list(self._histories):
            if key not in active_keys:
                del self._histories[key]

    def detect(self, frame: np.ndarray) -> Tuple[List[HandReading], Any]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        readings: List[HandReading] = []
        if not results.multi_hand_landmarks:
            self._histories.clear()
            return readings, results

        handedness_list = results.multi_handedness or []
        height, width = frame.shape[:2]
        active_keys: set[str] = set()

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_label = "Right"
            confidence = 0.0
            if idx < len(handedness_list):
                classification = handedness_list[idx].classification[0]
                mp_label = classification.label
                confidence = float(classification.score)

            hand_side = self.display_hand_side(mp_label)
            landmarks = hand_landmarks.landmark
            fingers = self.get_finger_states(landmarks, mp_label)
            hand_key = f"{hand_side}-{idx}"
            gesture = self.smooth_gesture(
                hand_key, self.recognize_gesture(fingers, landmarks)
            )
            bbox = self.hand_bbox(landmarks, width, height)
            active_keys.add(hand_key)

            readings.append(
                HandReading(
                    hand_side=hand_side,
                    confidence=confidence,
                    fingers=fingers,
                    gesture=gesture,
                    bbox=bbox,
                )
            )

        self.prune_histories(active_keys)
        return readings, results


def draw_panel(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int] = (12, 12),
    alpha: float = 0.55,
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    pad_x, pad_y, line_gap = 12, 10, 8

    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    text_h = max(h for _, h in sizes)
    box_w = max(w for w, _ in sizes) + pad_x * 2
    box_h = pad_y * 2 + len(lines) * text_h + (len(lines) - 1) * line_gap

    x0, y0 = origin
    x1, y1 = x0 + box_w, y0 + box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (24, 24, 24), -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (70, 70, 70), 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    y = y0 + pad_y + text_h
    for text in lines:
        cv2.putText(
            frame,
            text,
            (x0 + pad_x, y),
            font,
            scale,
            (240, 240, 240),
            thickness,
            cv2.LINE_AA,
        )
        y += text_h + line_gap


def draw_hand_overlay(frame: np.ndarray, reading: HandReading) -> None:
    x_min, y_min, x_max, y_max = reading.bbox
    color = (80, 180, 255) if reading.hand_side == "Right" else (80, 255, 160)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    label = f"{reading.hand_side} | {reading.gesture}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
    label_y = max(th + 8, y_min - 8)
    cv2.rectangle(
        frame,
        (x_min, label_y - th - 6),
        (x_min + tw + 10, label_y + 4),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x_min + 5, label_y),
        font,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    names = ("T", "I", "M", "R", "P")
    for i, (name, up) in enumerate(zip(names, reading.fingers)):
        cx = x_min + 14 + i * 22
        cy = min(frame.shape[0] - 10, y_max + 18)
        fill = color if up else (60, 60, 60)
        cv2.circle(frame, (cx, cy), 8, fill, -1)
        cv2.putText(
            frame,
            name,
            (cx - 4, cy + 4),
            font,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_help(frame: np.ndarray) -> None:
    help_lines = [
        "Controls",
        "q / Esc  Quit",
        "h        Toggle help",
        "d        Toggle landmarks",
        "m        Toggle mirror",
        "s        Screenshot",
    ]
    draw_panel(frame, help_lines, origin=(12, max(12, frame.shape[0] - 160)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam hand gesture detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--hands", type=int, default=2, choices=(1, 2), help="Max hands")
    parser.add_argument("--no-mirror", action="store_true", help="Disable selfie mirror")
    parser.add_argument(
        "--history",
        type=int,
        default=7,
        help="Gesture smoothing window size (frames)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_help = True
    show_landmarks = True

    detector = HandGestureDetector(
        max_num_hands=args.hands,
        mirror=not args.no_mirror,
        history_size=args.history,
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "Hand Gesture Detector"
    prev_time = time.perf_counter()
    fps = 0.0

    print("Hand Gesture Detector")
    print("Gestures: Fist, Thumbs Up, Point, Peace, Three, Four, Open Palm,")
    print("          Gun, OK, Call Me, Rock, Love You, Pinky")
    print("Press h for on-screen help, q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            if detector.mirror:
                frame = cv2.flip(frame, 1)

            readings, results = detector.detect(frame)

            if show_landmarks and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    detector.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        detector.mp_hands.HAND_CONNECTIONS,
                        detector.mp_styles.get_default_hand_landmarks_style(),
                        detector.mp_styles.get_default_hand_connections_style(),
                    )

            for reading in readings:
                draw_hand_overlay(frame, reading)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant = 1.0 / dt
                fps = 0.9 * fps + 0.1 * instant if fps > 0 else instant

            panel_lines = [
                f"FPS: {fps:4.1f}",
                f"Mirror: {'On' if detector.mirror else 'Off'}",
                f"Hands: {len(readings)}",
            ]
            if readings:
                for r in readings:
                    panel_lines.append(
                        f"{r.hand_side}: {r.gesture} ({r.confidence * 100:0.0f}%)"
                    )
                    panel_lines.append(
                        "  "
                        + " ".join(
                            n if up else "-"
                            for n, up in zip(("T", "I", "M", "R", "P"), r.fingers)
                        )
                    )
            else:
                panel_lines.append("No hand detected")

            draw_panel(frame, panel_lines)
            if show_help:
                draw_help(frame)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                show_help = not show_help
            elif key == ord("d"):
                show_landmarks = not show_landmarks
            elif key == ord("m"):
                detector.mirror = not detector.mirror
                detector._histories.clear()
            elif key == ord("s"):
                filename = f"gesture_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
