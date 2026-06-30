# Data Format

Data is stored as JSONL files in one directory. Each sample file is named:

```text
[id]_[easy|hard]_[alert|sleepy].jsonl
```

Examples:

```text
001_easy_alert.jsonl
002_hard_sleepy.jsonl
```

Each line is one frame.

## Easy Task Frame

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "gaze_screen_tf_calibrate_xy_px": [1268.4, 713.2],
  "target_xy_px": [1280, 720],
  "deviation_px_before_calibrate": 65.35,
  "deviation_px_after_calibrate": 13.19,
  "face_detection_bbox": [412, 216, 871, 799],
  "facial_landmark_35": [[520.0, 311.0], [541.0, 320.0]],
  "RetinaFace_bbox": [412, 216, 871, 799],
  "RetinaFace_landmarks": [[520.0, 311.0], [541.0, 320.0]],
  "confidence": 0.998
}
```

For easy samples, the per-frame input feature is:

```text
d_t = EuclideanDistance(gaze_screen_tf_calibrate_xy_px, target_xy_px)
```

If `gaze_screen_tf_calibrate_xy_px` is missing, the loader falls back to `gaze_screen_xy_px`.

## Hard Task Frame

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "gaze_screen_tf_calibrate_xy_px": [1268.4, 713.2],
  "target_centers_xy_px": [[1280, 720], [960, 540]],
  "deviation_px_before_calibrate": 67.12,
  "deviation_px_after_calibrate": 13.70,
  "face_detection_bbox": [412, 216, 871, 799],
  "facial_landmark_35": [[520.0, 311.0], [541.0, 320.0]],
  "RetinaFace_bbox": [412, 216, 871, 799],
  "RetinaFace_landmarks": [[520.0, 311.0], [541.0, 320.0]],
  "confidence": 0.998
}
```

For hard samples, the per-frame input feature is the nearest-anchor distance:

```text
d_t = min(EuclideanDistance(gaze_screen_tf_calibrate_xy_px, target) for target in target_centers_xy_px)
```
