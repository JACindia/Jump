import argparse
import cv2
import json
import math
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import requests
from ultralytics import YOLO


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Jump Action Detector - Only Show Jump BBOX")
    p.add_argument("--source", default="input.mp4", help="Video file path or webcam index")
    p.add_argument("--output", default="output.mp4", help="Output video path")
    p.add_argument("--model", default="yolov8m-pose.pt", help="YOLO pose model")
    p.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    p.add_argument("--no-window", action="store_true", help="Disable display window")
    p.add_argument("--no-telegram", action="store_true", help="Disable Telegram alerts")
    p.add_argument("--debug", action="store_true", help="Show debug overlay")
    return p.parse_args()


# =========================================================
# CONFIG
# =========================================================

class Config:
    MODEL_PATH = "yolov8m-pose.pt"
    TRACKER_CFG = "bytetrack.yaml"
    CONF_THRESHOLD = 0.35
    IOU_THRESHOLD = 0.50
    KEYPOINT_CONF_THR = 0.45

    MAX_MISSING_FRAMES = 40
    MIN_TRACK_AGE = 12

    GROUND_HISTORY = 40
    GROUND_PERCENTILE = 0.78
    FREEZE_GROUND_IN_AIR = True

    SMOOTH_WIN = 5
    MIN_VALID_BASELINE_FRAMES = 12
    MAX_POSE_MISS_STREAK = 8

    GROUND_MARGIN_NORM = 0.095
    LANDING_MARGIN_NORM = 0.055

    MIN_AIR_FRAMES = 4
    MIN_LAND_FRAMES = 4

    MIN_SEPARATION_PX = 8
    MAX_LEFT_RIGHT_DIFF_NORM = 0.22

    TAKEOFF_VEL_NORM = 0.009
    LANDING_VEL_NORM = 0.007
    HIP_LIFT_NORM = 0.018

    MAX_CENTER_DRIFT_NORM = 0.07

    MIN_TAKEOFF_SCORE = 7
    MIN_PEAK_SCORE = 7

    MIN_PEAK_HIP_GAP = 0.019
    MIN_PHYSICS_RATIO = 0.007
    MAX_PHYSICS_RATIO = 0.19

    ENABLE_TELEGRAM = True
    TELEGRAM_BOT_TOKEN = "8548372776:AAEgz4mOaDBU8HyrCklRHFHsnxT9U7HtRA0"
    TELEGRAM_CHAT_ID = "-5025882106"
    SEND_PHOTO = True
    TELEGRAM_COOLDOWN_SEC = 10

    ALERT_DIR = Path("jumps")


# COCO 17 keypoints
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


# =========================================================
# TELEGRAM
# =========================================================

def tg_send_message(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
        return r.ok
    except Exception as exc:
        print(f"[Telegram] message error: {exc}")
        return False


def tg_send_photo(token: str, chat_id: str, image_path: str, caption: str = "") -> bool:
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(image_path, "rb") as fh:
            r = requests.post(url, data={"chat_id": chat_id, "caption": caption},
                              files={"photo": fh}, timeout=20)
        return r.ok
    except Exception as exc:
        print(f"[Telegram] photo error: {exc}")
        return False


# =========================================================
# HELPERS
# =========================================================

def get_kpt(kpts: np.ndarray, idx: int, conf_thr: float):
    if kpts is None or idx >= len(kpts):
        return None
    row = kpts[idx]
    if len(row) < 2:
        return None
    x = float(row[0])
    y = float(row[1])
    c = float(row[2]) if len(row) > 2 else 1.0
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(c) or c < conf_thr:
        return None
    return x, y, c


def euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


def clip_box(x1, y1, x2, y2, W, H):
    return (
        max(0, min(int(x1), W - 1)),
        max(0, min(int(y1), H - 1)),
        max(0, min(int(x2), W - 1)),
        max(0, min(int(y2), W - 1)),
    )


def body_scale(kpts: np.ndarray, conf_thr: float, box=None):
    vals = []
    ls = get_kpt(kpts, KP_LEFT_SHOULDER, conf_thr)
    rs = get_kpt(kpts, KP_RIGHT_SHOULDER, conf_thr)
    if ls and rs:
        vals.append(euclidean(ls, rs))
    
    lh = get_kpt(kpts, KP_LEFT_HIP, conf_thr)
    rh = get_kpt(kpts, KP_RIGHT_HIP, conf_thr)
    if lh and rh:
        vals.append(euclidean(lh, rh))
    
    if box is not None:
        x1, y1, x2, y2 = box
        vals.append(max(1.0, float(y2 - y1) * 0.30))
    
    return max(vals) if vals else None


def safe_box_height(box):
    if box is None:
        return 100.0
    x1, y1, x2, y2 = box
    return max(1.0, float(y2 - y1))


# =========================================================
# GROUND ESTIMATOR
# =========================================================

class GroundEstimator:
    def __init__(self, maxlen: int = 40, percentile: float = 0.78):
        self._hist = deque(maxlen=maxlen)
        self._pct = percentile
        self._estimate = None
        self._frozen_value = None

    def update(self, y: float, frozen: bool = False):
        if y is None:
            return
        if not frozen:
            self._hist.append(float(y))
            self._estimate = self._compute()
            self._frozen_value = None
        else:
            if self._frozen_value is None and self._estimate is not None:
                self._frozen_value = self._estimate

    def _compute(self):
        if len(self._hist) < 12:
            return None
        vals = sorted(self._hist)
        cutoff = int(len(vals) * self._pct)
        keep = vals[cutoff:]
        return float(np.mean(keep)) if keep else None

    @property
    def value(self):
        return self._frozen_value if self._frozen_value is not None else self._estimate

    @property
    def ready(self):
        return self._estimate is not None

    @property
    def size(self):
        return len(self._hist)


# =========================================================
# PERSON STATE
# =========================================================

class PersonState:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.track_age = 0
        self.last_seen_frame = 0
        self.last_box = None
        self.pose_miss_streak = 0

        self.left_gnd = GroundEstimator()
        self.right_gnd = GroundEstimator()
        self.hip_gnd = GroundEstimator()

        self.left_y_hist = deque(maxlen=cfg.SMOOTH_WIN)
        self.right_y_hist = deque(maxlen=cfg.SMOOTH_WIN)
        self.hip_y_hist = deque(maxlen=cfg.SMOOTH_WIN)
        self.center_x_hist = deque(maxlen=cfg.SMOOTH_WIN)

        self.air_frames = 0
        self.land_frames = 0
        self.in_air = False
        self.jump_count = 0
        self.last_alert_t = 0.0
        self.alert_sent_this_jump = False
        self.best_air_score = -1.0
        self.best_air_image_path = None

        self.left_gap_norm = 0.0
        self.right_gap_norm = 0.0
        self.hip_gap_norm = 0.0
        self.left_vel_norm = 0.0
        self.right_vel_norm = 0.0
        self.hip_vel_norm = 0.0
        self.center_dx_norm = 0.0

        self.takeoff_started = False
        self.takeoff_frame = None
        self.peak_score = 0.0
        self.peak_hip_gap = 0.0
        self.peak_air_frames = 0

        self.debug_reasons = []

    def _smooth_and_vel(self, hist: deque, value: float):
        prev = float(np.median(hist)) if len(hist) else None
        hist.append(float(value))
        curr = float(np.median(hist))
        vel = 0.0 if prev is None else curr - prev
        return curr, vel

    def _reset_jump_window(self):
        self.air_frames = 0
        self.land_frames = 0
        self.takeoff_started = False
        self.takeoff_frame = None
        self.peak_score = 0.0
        self.peak_hip_gap = 0.0
        self.peak_air_frames = 0

    def _soft_decay(self):
        self.air_frames = max(0, self.air_frames - 1)
        self.land_frames = max(0, self.land_frames - 1)

    def update(self, kpts, frame_idx, annotated_frame, box, track_id, frame_wh):
        cfg = self.cfg
        self.track_age += 1
        self.last_seen_frame = frame_idx
        self.last_box = box
        self.debug_reasons = []

        la = get_kpt(kpts, KP_LEFT_ANKLE, cfg.KEYPOINT_CONF_THR)
        ra = get_kpt(kpts, KP_RIGHT_ANKLE, cfg.KEYPOINT_CONF_THR)
        lh = get_kpt(kpts, KP_LEFT_HIP, cfg.KEYPOINT_CONF_THR)
        rh = get_kpt(kpts, KP_RIGHT_HIP, cfg.KEYPOINT_CONF_THR)

        scale_px = body_scale(kpts, cfg.KEYPOINT_CONF_THR, box)
        if scale_px is None or scale_px < 10:
            scale_px = safe_box_height(box) * 0.30

        if la is None or ra is None or lh is None or rh is None:
            self.pose_miss_streak += 1
            self.debug_reasons.append("missing keypoints")
            if self.pose_miss_streak > cfg.MAX_POSE_MISS_STREAK:
                self._soft_decay()
                if not self.in_air:
                    self._reset_jump_window()
            else:
                self._soft_decay()
            return "Pose weak", (128, 128, 255)

        self.pose_miss_streak = 0

        if self.track_age < cfg.MIN_TRACK_AGE:
            return "Warming up", (180, 180, 0)

        hip_ctr = midpoint(lh, rh)
        if hip_ctr is None:
            self._soft_decay()
            return "Pose weak", (128, 128, 255)

        center_x = (la[0] + ra[0] + hip_ctr[0]) / 3.0

        left_y, left_vel = self._smooth_and_vel(self.left_y_hist, la[1])
        right_y, right_vel = self._smooth_and_vel(self.right_y_hist, ra[1])
        hip_y, hip_vel = self._smooth_and_vel(self.hip_y_hist, hip_ctr[1])
        _, center_dx = self._smooth_and_vel(self.center_x_hist, center_x)

        self.left_vel_norm = left_vel / scale_px
        self.right_vel_norm = right_vel / scale_px
        self.hip_vel_norm = hip_vel / scale_px
        self.center_dx_norm = abs(center_dx) / scale_px

        frozen = cfg.FREEZE_GROUND_IN_AIR and self.in_air
        self.left_gnd.update(left_y, frozen=frozen)
        self.right_gnd.update(right_y, frozen=frozen)
        self.hip_gnd.update(hip_y, frozen=frozen)

        if min(self.left_gnd.size, self.right_gnd.size, self.hip_gnd.size) < cfg.MIN_VALID_BASELINE_FRAMES:
            return "Calibrating", (180, 180, 0)

        left_gap_px = self.left_gnd.value - left_y
        right_gap_px = self.right_gnd.value - right_y
        hip_gap_px = self.hip_gnd.value - hip_y

        left_gap = left_gap_px / scale_px
        right_gap = right_gap_px / scale_px
        hip_gap = hip_gap_px / scale_px

        self.left_gap_norm = left_gap
        self.right_gap_norm = right_gap
        self.hip_gap_norm = hip_gap

        ankle_horizontal_dist = abs(la[0] - ra[0]) / scale_px if la and ra else 0.0
        too_wide_stance = ankle_horizontal_dist > 0.45

        ankle_air_ok = (
            left_gap > cfg.GROUND_MARGIN_NORM and
            right_gap > cfg.GROUND_MARGIN_NORM and
            left_gap_px > cfg.MIN_SEPARATION_PX and
            right_gap_px > cfg.MIN_SEPARATION_PX and
            not too_wide_stance
        )

        ankle_sym_ok = abs(left_gap - right_gap) <= cfg.MAX_LEFT_RIGHT_DIFF_NORM
        hip_lift_ok = hip_gap > cfg.HIP_LIFT_NORM

        upward_ok = (self.left_vel_norm < -cfg.TAKEOFF_VEL_NORM and 
                     self.right_vel_norm < -cfg.TAKEOFF_VEL_NORM)

        hip_upward_ok = self.hip_vel_norm < -(cfg.TAKEOFF_VEL_NORM * 0.45)
        low_horizontal_drift = self.center_dx_norm < cfg.MAX_CENTER_DRIFT_NORM

        jump_score = 0
        if ankle_air_ok:    jump_score += 3
        if ankle_sym_ok:    jump_score += 1
        if hip_lift_ok:     jump_score += 2
        if upward_ok:       jump_score += 2
        if hip_upward_ok:   jump_score += 1
        if low_horizontal_drift: jump_score += 1

        if not ankle_air_ok: self.debug_reasons.append("ankles low or wide stance")
        if not ankle_sym_ok: self.debug_reasons.append("ankle mismatch")
        if not hip_lift_ok:  self.debug_reasons.append("hip low")
        if not upward_ok:    self.debug_reasons.append("weak upward velocity")
        if not hip_upward_ok: self.debug_reasons.append("hip not rising")
        if not low_horizontal_drift: self.debug_reasons.append("horizontal drift")

        takeoff_candidate = jump_score >= cfg.MIN_TAKEOFF_SCORE

        landed_candidate = (
            left_gap < cfg.LANDING_MARGIN_NORM and
            right_gap < cfg.LANDING_MARGIN_NORM and
            self.left_vel_norm > cfg.LANDING_VEL_NORM and
            self.right_vel_norm > cfg.LANDING_VEL_NORM
        )

        if takeoff_candidate:
            self.air_frames += 1
            self.land_frames = 0
            if not self.takeoff_started:
                self.takeoff_started = True
                self.takeoff_frame = frame_idx
            self.peak_score = max(self.peak_score, float(jump_score))
            self.peak_hip_gap = max(self.peak_hip_gap, float(hip_gap))
            self.peak_air_frames = max(self.peak_air_frames, self.air_frames)

        elif landed_candidate:
            self.land_frames += 1
            self.air_frames = max(0, self.air_frames - 1)
        else:
            self._soft_decay()

        physics_ok = True
        if self.takeoff_started and self.peak_air_frames > 0:
            ratio = max(self.left_gap_norm, self.right_gap_norm, self.peak_hip_gap) / max(1, self.peak_air_frames)
            physics_ok = cfg.MIN_PHYSICS_RATIO <= ratio <= cfg.MAX_PHYSICS_RATIO
            if not physics_ok:
                self.debug_reasons.append("physics mismatch")

        confirmed_jump = (
            self.takeoff_started and
            self.air_frames >= cfg.MIN_AIR_FRAMES and
            self.peak_score >= cfg.MIN_PEAK_SCORE and
            self.peak_hip_gap >= cfg.MIN_PEAK_HIP_GAP and
            physics_ok and
            max(self.left_gap_norm, self.right_gap_norm) >= 0.07 and
            not too_wide_stance
        )

        state_text = "Grounded"
        color = (0, 165, 255)

        if confirmed_jump:
            self.in_air = True
            state_text = "JUMP ACTION"
            color = (0, 255, 0)

            current_score = min(left_gap, right_gap) + 0.5 * hip_gap
            if current_score > self.best_air_score:
                self.best_air_score = current_score
                W, H = frame_wh
                x1, y1, x2, y2 = clip_box(*box, W, H)
                snap = annotated_frame.copy()
                cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 255, 0), 4)
                _draw_text(snap, f"ID {track_id} JUMP ACTION", (x1, max(30, y1 - 42)), (0, 255, 0), 0.75, 2)
                _draw_text(snap, f"L:{left_gap:.3f} R:{right_gap:.3f} H:{hip_gap:.3f}", (x1, max(54, y1 - 14)), (255, 255, 255), 0.55, 2)
                path = cfg.ALERT_DIR / f"jump_id{track_id}_f{frame_idx}.jpg"
                cv2.imwrite(str(path), snap)
                self.best_air_image_path = str(path)

            now = time.time()
            if (cfg.ENABLE_TELEGRAM and not self.alert_sent_this_jump and self.best_air_image_path and
                cfg.TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN" and cfg.TELEGRAM_CHAT_ID != "YOUR_CHAT_ID" and
                (now - self.last_alert_t) >= cfg.TELEGRAM_COOLDOWN_SEC):
                self.last_alert_t = now
                self.alert_sent_this_jump = True
                self.jump_count += 1

                msg = f"JUMP ACTION DETECTED!\nID: {track_id}\nL: {left_gap:.3f} R: {right_gap:.3f} H: {hip_gap:.3f}"
                tg_send_message(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, msg)
                if cfg.SEND_PHOTO:
                    tg_send_photo(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, self.best_air_image_path,
                                  f"Jump | ID {track_id} | Score {self.best_air_score:.3f}")

        elif self.in_air and self.land_frames >= cfg.MIN_LAND_FRAMES:
            self.in_air = False
            self.alert_sent_this_jump = False
            self.best_air_score = -1.0
            self.best_air_image_path = None
            self._reset_jump_window()
            state_text = "Landed"
            color = (255, 0, 255)

        elif self.air_frames > 0:
            state_text = "Takeoff?"
            color = (0, 255, 255)
        else:
            if left_gap < 0.03 and right_gap < 0.03:
                self._reset_jump_window()

        return state_text, color


# =========================================================
# DRAWING
# =========================================================

def _draw_text(img, text, org, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def overlay_hud(frame, frame_idx: int, active_tracks: int):
    lines = [
        (f"Frame : {frame_idx}", (255, 255, 255)),
        (f"Tracks: {active_tracks}", (200, 200, 200)),
    ]
    for i, (txt, col) in enumerate(lines):
        _draw_text(frame, txt, (16, 30 + i * 26), col, 0.72, 2)


def draw_person(frame, track_id, box, state_text, color, person: PersonState, frame_wh, debug=False):
    """Only draw bounding box when it's JUMP ACTION"""
    if state_text != "JUMP ACTION":
        return

    W, H = frame_wh
    x1, y1, x2, y2 = clip_box(*box, W, H)

    # Thick green box for jump only
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Label
    _draw_text(frame, f"ID {track_id} JUMP ACTION", 
               (x1, max(25, y1 - 55)), (0, 255, 0), 0.75, 2)

    # Gap values
    _draw_text(frame, f"L:{person.left_gap_norm:.3f} R:{person.right_gap_norm:.3f} H:{person.hip_gap_norm:.3f}",
               (x1, max(50, y1 - 25)), (255, 255, 255), 0.55, 2)

    if debug:
        for i, reason in enumerate(person.debug_reasons[:8]):
            yy = min(H - 5, y2 + 18 + i * 18)
            _draw_text(frame, f"- {reason}", (x1, yy), (80, 220, 255), 0.42, 1)


# =========================================================
# STATS
# =========================================================

def save_stats(cfg: Config, track_states: dict, frame_count: int):
    stats = {"frame_count": frame_count, "total_jumps": int(sum(p.jump_count for p in track_states.values()))}
    path = cfg.ALERT_DIR / "jump_stats.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)


# =========================================================
# MAIN
# =========================================================

def run(cfg: Config):
    cfg.ALERT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Init] Loading model: {cfg.MODEL_PATH}")
    model = YOLO(cfg.MODEL_PATH)

    source = int(cfg.VIDEO_SOURCE) if str(cfg.VIDEO_SOURCE).isdigit() else cfg.VIDEO_SOURCE
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {cfg.VIDEO_SOURCE!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_wh = (frame_w, frame_h)

    print(f"[Init] Source: {cfg.VIDEO_SOURCE} {frame_w}x{frame_h} {fps:.1f} fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cfg.OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))

    track_states = defaultdict(lambda: PersonState(cfg))
    frame_idx = 0

    print("[Run] Processing... Press Q to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Run] End of stream.")
                break

            frame_idx += 1

            result = model.track(
                frame, persist=True, tracker=cfg.TRACKER_CFG,
                conf=cfg.CONF_THRESHOLD, iou=cfg.IOU_THRESHOLD,
                verbose=False, device=cfg.DEVICE, classes=[0]
            )[0]

            annotated = frame.copy()

            has_tracks = (result.boxes is not None and 
                         getattr(result.boxes, "id", None) is not None and
                         result.keypoints is not None and len(result.boxes) > 0)

            if has_tracks:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                ids = result.boxes.id.int().detach().cpu().tolist()
                kpts_all = result.keypoints.data.detach().cpu().numpy()

                n = min(len(boxes), len(ids), len(kpts_all))
                for i in range(n):
                    track_id = ids[i]
                    person = track_states[track_id]
                    kpts = kpts_all[i]
                    box = boxes[i]

                    state_text, color = person.update(kpts, frame_idx, annotated, box, track_id, frame_wh)

                    # Only draw when it's a confirmed JUMP ACTION
                    draw_person(annotated, track_id, box, state_text, color, person, frame_wh, debug=cfg.DEBUG)

            # Remove stale tracks
            stale = [tid for tid, p in track_states.items() 
                     if frame_idx - p.last_seen_frame > cfg.MAX_MISSING_FRAMES]
            for tid in stale:
                del track_states[tid]

            total_jumps = sum(p.jump_count for p in track_states.values())
            overlay_hud(annotated, frame_idx, len(track_states))

            writer.write(annotated)

            if cfg.SHOW_WINDOW:
                cv2.imshow("Jump Action Detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Run] User quit.")
                    break

    except KeyboardInterrupt:
        print("\n[Run] Interrupted.")
    except Exception:
        traceback.print_exc()
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        save_stats(cfg, track_states, frame_idx)

    print(f"\n[Done] Output   -> {cfg.OUTPUT_PATH}")
    print(f"[Done] Frames   : {frame_idx}")


def main():
    args = parse_args()
    cfg = Config()
    cfg.MODEL_PATH = args.model
    cfg.VIDEO_SOURCE = args.source
    cfg.OUTPUT_PATH = args.output
    cfg.DEVICE = args.device
    cfg.SHOW_WINDOW = not args.no_window
    cfg.ENABLE_TELEGRAM = not args.no_telegram
    cfg.DEBUG = args.debug

    run(cfg)


if __name__ == "__main__":
    main()