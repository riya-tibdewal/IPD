# EduAware Optimization Notes

## Runtime Defaults

- `FRAME_SKIP = 2`: pose inference runs every second frame by default.
- Rendering still runs every frame using cached per-track state for smoother UI.
- `EMA_ALPHA = 0.15`: attention and class scores change smoothly.
- `HYSTERESIS_MARGIN = 0.15`: labels switch only when the new class is clearly stronger.
- `STALE_TRACK_S = 7.0`: keeps IDs stable through short occlusions.
- `YOLO_IMGSZ = 480`, `YOLO_CONF = 0.45`, `KP_CONF = 0.30`: balanced CPU/GPU defaults.

## Behavior Stability

- Each student keeps 60 frames of temporal memory.
- Strong evidence can switch faster, but noisy evidence waits longer.
- Writing now requires desk-hand context plus writing-object or motion/back-view evidence.
- Sleeping suppresses writing/standing when head-down slouch and stillness persist.
- Standing requires visible legs or a true full-body/tall pose, not just a seated tall crop.
- Talking is restored as a separate class for side/peer-facing interaction.

## Performance Monitoring

The HUD now displays:

- FPS
- pose inference milliseconds
- object detection milliseconds
- draw/render milliseconds
- skipped/dropped inference frames

## Object Association

When `--objects` is enabled, YOLO object detections are grouped into:

- phone cues: `cell phone`, `mobile phone`, `phone`
- writing cues: `book`, `notebook`, `paper`, `laptop`, `keyboard`

Objects are associated with the nearest/overlapping student box.

## Recommended Commands

CPU:

```powershell
python main.py --src 0 --objects
```

Faster CPU:

```powershell
python main.py --src 0 --imgsz 416 --skip 2
```

GPU:

```powershell
python main.py --src 0 --objects --gpu --model yolov8s-pose.pt --imgsz 640
```

Video file:

```powershell
python main.py --src "path\to\classroom.mp4" --objects
```

