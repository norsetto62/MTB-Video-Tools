#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import cv2

from autocut.utils import convert_s_to_hms
from autocut.annotations import load_training_annotations
from autocut.video.probe import get_video_info
from autocut.video.reader import VideoReader
from autocut.motion.flow import calculate_optical_flow
from autocut.motion.features import FeatureExtractor

SAMPLE_FPS = 2.0
PROCESSING_WIDTH = 480

def parse_arguments():

    parser = argparse.ArgumentParser(
                        prog='audit_dataset',
                        description='Data audit for the MTB autocut project',
                        epilog='(c)2026 Anziano in (e)Bicicletta',
                        usage='%(prog)s input [options]')

    parser.add_argument('input', type=Path, help='annotations file absolute path')
    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"Error: input video does not exist:\n{args.input}",
            file=sys.stderr,
        )
        sys.exit(1)
    
    return args.input

def main()->int:

    annotation_path = parse_arguments()
    output_dir = Path("audit") / annotation_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Processing ", annotation_path)

    video_path, annotations = load_training_annotations(annotation_path)
    video_info = get_video_info(video_path)

    #We scale down to PROCESSING_WIDTH
    processing_width = PROCESSING_WIDTH
    processing_height = round(video_info.height * processing_width / video_info.width)
    if processing_height % 2:
        processing_height += 1

    #We crop 15% left/right, 10% top and 50% botton
    roi_x0 = round(processing_width * 0.15)
    roi_width = round(processing_width * 0.7)
    roi_y0 = round(processing_height * 0.1)
    roi_height = round(processing_height * 0.4)

    index = 0
    total_frames = 0
    tot_annotations = len(annotations)
    eta = 0
    processing_start_wall = cv2.getTickCount()

    for annotation in annotations:
        features = []
        timestamps = []
        index += 1
        with VideoReader(video_path=video_path, start=annotation.start, duration=annotation.end - annotation.start, sample_fps=SAMPLE_FPS) as training_video:

            progress = training_video.start*100/video_info.duration
            elapsed_wall = (
                cv2.getTickCount() -
                processing_start_wall
            ) / cv2.getTickFrequency()
            if progress > 0:
                eta = elapsed_wall*(100 - progress)/progress
                print(f"Progress: {progress:5.2f}% processed annotations {index-1} of {tot_annotations} in {convert_s_to_hms(elapsed_wall)}, eta {convert_s_to_hms(eta)}")
            previous_gray=None
            extractor=FeatureExtractor(roi_width=roi_width, roi_height=roi_height)
            for frame_index, timestamp, frame in training_video:

                frame = cv2.resize(
                    frame,
                    (processing_width, processing_height),
                    interpolation=cv2.INTER_AREA,
                )
                gray = frame[roi_y0:roi_y0 + roi_height, roi_x0:roi_x0 + roi_width]
                current_gray = cv2.cvtColor(
                    gray,
                    cv2.COLOR_BGR2GRAY,
                )
                if previous_gray is not None:
                    flow = calculate_optical_flow(previous_gray=previous_gray, current_gray=current_gray)
                    features.append(extractor.extract_vector(flow))
                    timestamps.append(timestamp)
                    total_frames += 1
                previous_gray = current_gray
        filename = f"annotation_{index:03d}.npz"
        npz_path = output_dir / filename        
        np.savez(
            npz_path,
            features=np.asarray(features, dtype=np.float32),
            timestamps=np.asarray(timestamps, dtype=np.float64),
        )

    elapsed_wall = (
        cv2.getTickCount() -
        processing_start_wall
    ) / cv2.getTickFrequency()
    print(f"Processed {total_frames} frames in {convert_s_to_hms(elapsed_wall)}")
    print(f"Data saved in {output_dir}")
    
    return 0
"""
    sequence = build_feature_sequence(
        video_path,
        ...
    )

    audit_feature_sequence(sequence)
"""

if __name__ == "__main__":
    raise SystemExit(main())