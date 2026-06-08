"""
Generate episode frame index npz file for Vitra dataset.

This script recursively finds all annotation files in the format of
{dataset_name}_{video_name}_ep_000000.npy and generates an episode_frame_index.npz
file containing:
- index_frame_pair: [ep_index, frame_index] pairs for each sample
- index_to_episode_id: mapping from ep_index to readable episode identifier
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# Pattern to match: {dataset_name}_{video_name}_ep_XXXXXX.npy
# The episode number is always 6 digits
ANNOTATION_FILE_PATTERN = re.compile(r'^(.+_.+)_ep_(\d{6})\.npy$')


def find_annotation_files(base_path: str) -> List[str]:
    """
    Recursively find all annotation files matching the pattern.

    Args:
        base_path: Base directory path to search in

    Returns:
        List of absolute paths to annotation files
    """
    base_path = Path(base_path)
    annotation_files = []

    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        for file in files:
            if file.endswith('.npy') and '_ep_' in file:
                # Check if it matches the expected pattern
                match = ANNOTATION_FILE_PATTERN.match(file)
                if match:
                    annotation_files.append(str(root_path / file))

    return sorted(annotation_files)


def parse_episode_info(file_path: str) -> Tuple[str, str, int]:
    """
    Parse episode information from file path.

    Args:
        file_path: Absolute path to the annotation file

    Returns:
        Tuple of (full_episode_name, dataset_name, episode_index)
        - full_episode_name: e.g., "WorldModelV1_gpu0_task000000_ep_000000"
        - dataset_name: e.g., "WorldModelV1_gpu0_task000000"
        - episode_index: integer episode number
    """
    filename = os.path.basename(file_path)
    match = ANNOTATION_FILE_PATTERN.match(filename)
    if not match:
        raise ValueError(f"File does not match expected pattern: {file_path}")

    dataset_name = match.group(1)
    episode_num_str = match.group(2)
    episode_index = int(episode_num_str)

    # Full episode name without .npy extension
    full_episode_name = f"{dataset_name}_ep_{episode_num_str}"

    return full_episode_name, dataset_name, episode_index


def generate_episode_index(
    annotation_files: List[str],
    start_index: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate index frame pairs and episode ID mappings.

    Args:
        annotation_files: List of paths to annotation files
        start_index: Starting index for frame numbering

    Returns:
        Tuple of (index_frame_pair, index_to_episode_id)
        - index_frame_pair: np.ndarray of shape (N, 2) where N is total frames
        - index_to_episode_id: np.ndarray of episode ID strings
    """
    if not annotation_files:
        # Return empty arrays if no files found
        return np.array([], dtype=np.int64).reshape(0, 2), np.array([], dtype=object)

    # Build episode index mapping
    episode_id_list = []
    episode_to_index = {}  # episode_name -> ep_index
    episode_frames = {}  # episode_name -> number of frames

    for file_path in annotation_files:
        full_name, dataset_name, episode_idx = parse_episode_info(file_path)

        # Load the npy file to get the number of frames
        # npy file is a dictionary with 'video_decode_frame' containing frame indices
        try:
            data = np.load(file_path, allow_pickle=True)
            if data.ndim == 0 and data.item() is not None:
                data_dict = data.item()
                num_frames = len(data_dict.get('video_decode_frame', []))
            else:
                num_frames = len(data)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            num_frames = 0

        if full_name not in episode_to_index:
            ep_index = len(episode_id_list)
            episode_to_index[full_name] = ep_index
            episode_id_list.append(full_name)
            episode_frames[full_name] = num_frames
        else:
            # If same episode appears multiple times, take the maximum frame count
            episode_frames[full_name] = max(episode_frames[full_name], num_frames)

    # Build index_frame_pair
    total_frames = sum(episode_frames.values())
    index_frame_pair = np.zeros((total_frames, 2), dtype=np.int64)

    current_frame = start_index
    for ep_name in episode_id_list:
        ep_idx = episode_to_index[ep_name]
        num_frames = episode_frames[ep_name]

        for frame_offset in range(num_frames):
            index_frame_pair[current_frame, 0] = ep_idx
            index_frame_pair[current_frame, 1] = frame_offset
            current_frame += 1

    # Build index_to_episode_id
    index_to_episode_id = np.array(episode_id_list, dtype=object)

    return index_frame_pair, index_to_episode_id


def generate_episodic_index(
    base_path: str,
    output_filename: str = "episode_frame_index.npz",
    verbose: bool = True
) -> Optional[str]:
    """
    Main function to generate the episode frame index.

    Args:
        base_path: Base directory path containing annotation files
        output_filename: Name of the output npz file
        verbose: Whether to print progress information

    Returns:
        Path to the generated npz file, or None if no files found
    """
    base_path = Path(base_path)

    if verbose:
        print(f"Scanning for annotation files in: {base_path}")

    # Find all annotation files
    annotation_files = find_annotation_files(str(base_path))

    if not annotation_files:
        if verbose:
            print("No annotation files found matching the pattern.")
        return None

    if verbose:
        print(f"Found {len(annotation_files)} annotation files")

    # Generate index data
    index_frame_pair, index_to_episode_id = generate_episode_index(annotation_files)

    # Save to npz file
    output_path = base_path / output_filename
    np.savez(
        output_path,
        index_frame_pair=index_frame_pair,
        index_to_episode_id=index_to_episode_id
    )

    if verbose:
        print(f"Generated: {output_path}")
        print(f"  - Total frames: {len(index_frame_pair)}")
        print(f"  - Total episodes: {len(index_to_episode_id)}")
        print(f"  - index_frame_pair shape: {index_frame_pair.shape}")
        print(f"  - index_to_episode_id shape: {index_to_episode_id.shape}")

    return str(output_path)


def verify_episode_index(annotation_file: str) -> bool:
    """
    Verify the generated episode index file.

    Args:
        annotation_file: Path to the episode_frame_index.npz file

    Returns:
        True if the file is valid and consistent
    """
    try:
        data = np.load(annotation_file, allow_pickle=True)
        index_frame_pair = data['index_frame_pair']
        index_to_episode_id = data['index_to_episode_id']

        # Check consistency
        max_ep_idx = index_frame_pair[:, 0].max() if len(index_frame_pair) > 0 else -1
        expected_max_ep_idx = len(index_to_episode_id) - 1

        if max_ep_idx != expected_max_ep_idx:
            print(f"Warning: Max ep_idx ({max_ep_idx}) != expected ({expected_max_ep_idx})")
            return False

        # Check frame indices are non-negative and within expected range
        frame_indices = index_frame_pair[:, 1]
        if (frame_indices < 0).any():
            print("Warning: Found negative frame indices")
            return False

        print("Verification passed!")
        print(f"  - Total episodes: {len(index_to_episode_id)}")
        print(f"  - Total frames: {len(index_frame_pair)}")
        print(f"  - First 5 index_frame_pair entries:")
        for i in range(min(5, len(index_frame_pair))):
            print(f"    {index_frame_pair[i]}")
        print(f"  - Last 5 index_frame_pair entries:")
        for i in range(max(0, len(index_frame_pair) - 5), len(index_frame_pair)):
            print(f"    {index_frame_pair[i]}")
        print(f"  - Sample episode IDs: {index_to_episode_id[:3]} ... {index_to_episode_id[-3:]}")

        return True

    except Exception as e:
        print(f"Error verifying file: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate episode frame index for Vitra dataset"
    )
    parser.add_argument(
        "base_path",
        help="Base directory path containing annotation files"
    )
    parser.add_argument(
        "--output", "-o",
        default="episode_frame_index.npz",
        help="Output filename (default: episode_frame_index.npz)"
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify the generated index file"
    )

    args = parser.parse_args()

    if args.verify:
        verify_episode_index(args.base_path)
    else:
        generate_episodic_index(args.base_path, args.output)
