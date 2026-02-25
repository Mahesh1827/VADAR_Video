# VADAR Video Analysis Extension

Extend VADAR from single-image visual reasoning to **video analysis** by unlocking SAM 2's built-in Video Predictor, adding temporal modules, and adapting the LLM prompt pipeline.

## User Review Required

> [!IMPORTANT]
> **Hardware Requirements**: SAM 2 Video Mode requires **≥16 GB VRAM**. A chunked processing fallback (50 frames at a time) will be implemented for lower-VRAM GPUs.

> [!IMPORTANT]
> **Scope**: This adds video as a **new mode** alongside the existing image mode. All existing image functionality remains untouched.

> [!WARNING]
> **Dependency**: The `sam2` library must include `sam2.sam2_video_predictor`. This will be verified at import time.

## Proposed Changes

### Video Infrastructure

#### [NEW] [video_utils.py](file:///c:/Users/Admin/VADAR_Video/engine/video_utils.py)
- `extract_frames`: Extracts frames from video to JPEG directory.
- `get_video_metadata`: Resolution, frames, FPS, duration.
- `chunk_frame_indices`: Splits indices for low-VRAM processing.
- `frames_to_video`: Re-assembles frames into video.

#### [NEW] [video_engine.py](file:///c:/Users/Admin/VADAR_Video/engine/video_engine.py)
- `VideoEngine`: Extends execution model to handle `video_dir`, `frame_count`, and `fps`.

### Core Logic Extensions

#### [MODIFY] [predefined_modules.py](file:///c:/Users/Admin/VADAR_Video/engine/predefined_modules.py)
- `VideoTrackModule`: Uses `SAM2VideoPredictor` to track objects across frames.
- `VideoLocateModule`: Runs `LocateModule` on individual frames.
- `VideoVQAModule`: Runs VQA on specific frames.
- `VideoResultModule`: Aggregates temporal data.
- `VideoModulesList`: Initializes SAM 2 in video mode.

### Prompt Engineering

#### [MODIFY] [modules.py](file:///c:/Users/Admin/VADAR_Video/prompts/modules.py)
- Add `VIDEO_MODULES_SIGNATURES` for video-specific API functions (`track`, `loc_frame`, etc.).

#### [MODIFY] [program_prompt.py](file:///c:/Users/Admin/VADAR_Video/prompts/program_prompt.py)
- Add `VIDEO_PROGRAM_PROMPT` with video-specific few-shot examples.

## Verification Plan

### Automated Verification
- **Import Check**: Run `python -c "from engine.video_engine import VideoEngine; from engine.predefined_modules import VideoModulesList; print('OK')"` to verify imports.

### Manual Verification
- **Frame Extraction**: Verify frames are saved as JPEGs from a sample video.
- **End-to-End**: Test with a simple question (e.g., "How many people are in the video?") and verify the generated program and answer.
- **Chunked Processing**: Verify fallback works if VRAM < 16GB.
