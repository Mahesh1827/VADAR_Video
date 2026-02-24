"""
Video utility functions for VADAR video analysis.

Provides frame extraction, video metadata retrieval, chunked processing
for memory management, and frame reassembly into output videos.
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image


def extract_frames(video_path, output_dir, fps=None):
    """
    Extract frames from a video file and save as JPEG images.

    SAM 2 Video Predictor requires a directory of JPEG frames as input.
    This function extracts frames at the original FPS or a specified FPS.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted JPEG frames.
        fps (float, optional): Target frames per second. If None, uses
            the video's native FPS (extracts every frame).

    Returns:
        dict: Metadata about the extracted video:
            - 'frame_count': Number of extracted frames
            - 'original_fps': Original video FPS
            - 'target_fps': FPS used for extraction
            - 'width': Frame width in pixels
            - 'height': Frame height in pixels
            - 'duration': Video duration in seconds
            - 'output_dir': Path to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / original_fps if original_fps > 0 else 0

    # Calculate frame sampling interval
    if fps is None or fps >= original_fps:
        target_fps = original_fps
        frame_interval = 1
    else:
        target_fps = fps
        frame_interval = original_fps / fps

    frame_idx = 0
    saved_count = 0
    next_sample = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= next_sample:
            # Save frame as JPEG with zero-padded filename
            frame_path = os.path.join(output_dir, f"{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            next_sample += frame_interval

        frame_idx += 1

    cap.release()

    return {
        "frame_count": saved_count,
        "original_fps": original_fps,
        "target_fps": target_fps,
        "width": width,
        "height": height,
        "duration": duration,
        "output_dir": output_dir,
    }


def get_video_metadata(video_path):
    """
    Get metadata about a video file without extracting frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: Video metadata:
            - 'fps': Frames per second
            - 'total_frames': Total number of frames
            - 'width': Frame width in pixels
            - 'height': Frame height in pixels
            - 'duration': Duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


def chunk_frame_indices(total_frames, chunk_size=50):
    """
    Split frame indices into chunks for memory-efficient processing.

    When GPU VRAM is limited, SAM 2 Video Predictor can process frames
    in chunks instead of loading the entire video context at once.

    Args:
        total_frames (int): Total number of frames.
        chunk_size (int): Maximum frames per chunk. Defaults to 50.

    Returns:
        list[tuple]: List of (start_idx, end_idx) tuples. Each tuple
            represents an inclusive range of frame indices.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks = []
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size - 1, total_frames - 1)
        chunks.append((start, end))
    return chunks


def load_frame(video_dir, frame_index):
    """
    Load a single frame from the extracted frames directory as a PIL Image.

    Args:
        video_dir (str): Path to directory containing extracted JPEG frames.
        frame_index (int): Zero-based index of the frame to load.

    Returns:
        PIL.Image: The frame as an RGB PIL Image.
    """
    frame_path = os.path.join(video_dir, f"{frame_index:06d}.jpg")
    if not os.path.exists(frame_path):
        raise FileNotFoundError(
            f"Frame {frame_index} not found at {frame_path}"
        )
    return Image.open(frame_path).convert("RGB")


def get_frame_count(video_dir):
    """
    Count the number of extracted frames in a directory.

    Args:
        video_dir (str): Path to directory containing extracted JPEG frames.

    Returns:
        int: Number of JPEG frame files found.
    """
    pattern = os.path.join(video_dir, "*.jpg")
    return len(glob.glob(pattern))


def frames_to_video(frame_dir, output_path, fps=30.0):
    """
    Reassemble JPEG frames from a directory into a video file.

    Useful for creating annotated output videos with tracking overlays.

    Args:
        frame_dir (str): Directory containing JPEG frames (zero-padded names).
        output_path (str): Path for the output video file (.mp4).
        fps (float): Frames per second for the output video. Defaults to 30.

    Returns:
        str: Path to the created video file.
    """
    pattern = os.path.join(frame_dir, "*.jpg")
    frame_files = sorted(glob.glob(pattern))

    if not frame_files:
        raise ValueError(f"No JPEG frames found in {frame_dir}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        writer.write(frame)

    writer.release()
    return output_path


def bbox_from_mask(mask, margin=10):
    """
    Compute a bounding box from a binary segmentation mask.

    Args:
        mask (np.ndarray): 2D binary mask (H, W).
        margin (int): Pixel margin to add around the bounding box.

    Returns:
        list: [x_min, y_min, x_max, y_max] or None if mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add margin
    rmin = max(0, rmin - margin)
    cmin = max(0, cmin - margin)
    rmax = min(mask.shape[0] - 1, rmax + margin)
    cmax = min(mask.shape[1] - 1, cmax + margin)

    return [cmin, rmin, cmax, rmax]
