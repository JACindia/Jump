# 🏃‍♂️ Jump Action Detection System

> Real-time human jump detection using **YOLOv8 Pose Estimation** + **ByteTrack** tracking — with Telegram alerts, jump snapshots, and output video generation.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🎯 Real-time detection | Live jump detection from webcam or video |
| 🧠 Pose-based analysis | Uses ankles, hips, and shoulders keypoints |
| 📊 Physics validation | Multi-condition motion + velocity checks |
| 🟩 Smart bounding box | Shown **only during confirmed jump** |
| 📸 Frame capture | Saves best jump frame per person |
| 📩 Telegram alerts | Sends image + message with cooldown |
| 📈 Jump counting | Tracks total jumps across session |
| 🎥 Video output | Saves annotated output video |
| 🧪 Debug mode | Verbose condition logging |

---

## 🧰 Tech Stack

- **Python 3.8+**
- **OpenCV** — Video I/O and frame rendering
- **NumPy** — Keypoint math and array ops
- **Ultralytics YOLOv8** — Pose estimation model
- **ByteTrack** — Multi-person tracking with IDs
- **Requests** — Telegram Bot API integration


---

## ⚙️ Installation

```bash
pip install ultralytics opencv-python numpy requests
```

---

## ▶️ Usage

```bash
# Run with a video file
python main.py --source input.mp4 --output output.mp4

# Run with webcam
python main.py --source 0

# Enable debug mode
python main.py --debug

# Disable the display window
python main.py --no-window

# Disable Telegram alerts
python main.py --no-telegram
```

---

## 🧠 How It Works

### 1. Pose Detection
YOLOv8 detects 17 body keypoints per person each frame, including ankles, knees, hips, and shoulders.

### 2. Tracking
ByteTrack assigns stable unique IDs to each person across frames, even through occlusion.

### 3. Ground Estimation
A rolling baseline is maintained per person to estimate their typical standing ankle height.

### 4. Jump Confirmation
A jump is confirmed when **all** of the following are true:

- ✅ Ankles are above the ground baseline
- ✅ Hip is lifted relative to ground
- ✅ Upward velocity is detected
- ✅ Motion is vertically symmetric
- ✅ Horizontal drift is within limits
- ✅ Physics conditions are valid

---

## ⚙️ Configuration

Tune these parameters in the `Config` class inside `main.py`:

| Parameter | Description | Default |
|---|---|---|
| `CONF_THRESHOLD` | Minimum detection confidence | `0.5` |
| `GROUND_MARGIN_NORM` | Normalized air gap to count as airborne | `0.05` |
| `TAKEOFF_VEL_NORM` | Minimum upward velocity to confirm jump | `0.02` |
| `MIN_AIR_FRAMES` | Minimum consecutive frames in air | `3` |
| `MAX_CENTER_DRIFT_NORM` | Max horizontal drift allowed | `0.15` |

---

## 📸 Output

- **Green bounding box** appears around the person only during a confirmed jump
- **Jump frames** are saved to `/jumps/jump_idX_fY.jpg`
- **Output video** is saved as `output.mp4`
- **Session stats** are written to `jump_stats.json`:

```json
{
  "frame_count": 1200,
  "total_jumps": 5
}
```

---

## 📩 Telegram Setup

Edit the following in the `Config` class:

```python
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID   = "your_chat_id"
```

The bot will:
- Send an alert message when a jump is detected
- Attach the best captured jump frame
- Enforce a cooldown period to prevent alert spam

To get your credentials:
1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Grab your `bot_token` from the response
3. Get your `chat_id` by messaging your bot and calling `getUpdates`

---

## 🧪 Debug Mode

Run with `--debug` to enable verbose output:

```
[ID 1] ankle_lift=0.08 hip_lift=0.06 vel=0.031 drift=0.04 → JUMP CONFIRMED
[ID 2] ankle_lift=0.02 → REJECTED: below ground threshold
```

Shows per-frame:
- Detection conditions and values
- Motion validation results
- Rejection reasons

---

## ⚠️ Limitations

- Requires **full body visibility** in frame
- Sensitive to **occlusion** by other people or objects
- Performance degrades under **poor lighting** conditions
- Designed for **single or sparse multi-person** scenes

---

## 📈 Future Improvements

- [ ] Multi-action recognition (squat, sprint, fall)
- [ ] Edge deployment (Jetson Nano / Raspberry Pi)
- [ ] Web dashboard with live stats
- [ ] Lightweight model optimization (ONNX / TensorRT)
- [ ] REST API for integration with other systems

---

## 👨‍💻 Author

Built with computer vision and real-time AI for sports analytics, fitness tracking, and surveillance applications.
