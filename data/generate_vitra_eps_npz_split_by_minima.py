"""
根据速度极小值分割episode，生成split版本的npy文件和分段信息json。

规则：
- 排除极小值>2的eps
- 1个极小值：2段 -> 原指令前半（或整句）+ "return home"
- 2个极小值：3段 -> 原指令按 ` and ` 拆分后的前两段 + "return home"（无下半句时中段为 "lift it"）

每个npy文件分割后单独保存到对应的 episodic_annotations_split 目录。
同时生成一个 json 文件记录所有分段信息。
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """支持 numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


ANNOTATION_FILE_PATTERN = re.compile(r'^(.+)_ep_(\d{6})\.npy$')

# 右手有效帧过滤参数
RIGHT_HAND_THRESHOLD = 0.5  # 有效帧比例阈值


def check_right_hand_ratio(file_path: str, threshold: float = 0.5) -> Tuple[bool, float, int, int]:
    """
    检查右手有效帧比例

    Args:
        file_path: npy文件路径
        threshold: 有效帧最小比例 (默认0.5=50%)

    Returns:
        Tuple of (is_valid, ratio, valid_frames, total_frames)
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        if data.ndim == 0 and data.item() is not None:
            data_dict = data.item()
        else:
            data_dict = data

        if 'right' not in data_dict:
            return False, 0.0, 0, 0

        right_data = data_dict['right']
        if not isinstance(right_data, dict) or 'kept_frames' not in right_data:
            return False, 0.0, 0, 0

        kept_frames = right_data['kept_frames']
        if not isinstance(kept_frames, (list, np.ndarray)):
            return False, 0.0, 0, 0

        kept_frames = np.array(kept_frames)
        total_frames = len(kept_frames)
        valid_frames = int(np.sum(kept_frames))

        if total_frames == 0:
            return False, 0.0, 0, 0

        ratio = valid_frames / total_frames
        is_valid = ratio >= threshold

        return is_valid, ratio, valid_frames, total_frames

    except Exception as e:
        return False, 0.0, 0, 0

# 速度极小值检测参数
MIN_FRAMES = 15  # 前后各15帧不算
SPEED_THRESHOLD = 0.01  # 极小值速度要小于这个阈值
PROMINENCE = 0.002  # 极小值的下陷深度
MIN_DISTANCE = 8  # 极小值之间的最小帧距
FLAT_WINDOW = 10  # 平坦度检测窗口
# 极值点位置约束（宽松范围）
MINIMA_POS_1_MIN, MINIMA_POS_1_MAX = 15, 35  # 第一个极值点范围
MINIMA_POS_2_MIN, MINIMA_POS_2_MAX = 50, 70  # 第二个极值点范围


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """使用滑动均值平滑"""
    return np.convolve(x, np.ones(window) / window, mode='same')


def find_speed_minima(speed: np.ndarray) -> np.ndarray:
    """
    找速度极小值点，返回索引数组
    
    策略：优先从指定位置范围中选择速度最小且平坦的点
    """
    from scipy.signal import find_peaks
    
    T = len(speed)
    speed_smooth = smooth(speed)
    neg_speed = -speed_smooth
    peaks, _ = find_peaks(neg_speed, prominence=PROMINENCE, distance=MIN_DISTANCE)
    
    # 过滤：极小值速度要足够小
    valid_peak = peaks[speed_smooth[peaks] < SPEED_THRESHOLD]
    
    # 只保留中间区域
    mid_mask = (valid_peak >= MIN_FRAMES) & (valid_peak < T - MIN_FRAMES)
    mid_peaks = valid_peak[mid_mask]
    
    if len(mid_peaks) == 0:
        return np.array([], dtype=int)
    
    # 选第一个和最后一个
    min_idx = np.array([mid_peaks[0], mid_peaks[-1]], dtype=int)
    
    # 有效范围修正：+5 后夹紧到 [0, T-1]
    min_idx = np.clip(min_idx + 5, 0, T - 1)
    
    return min_idx


def extract_right_hand_speed(data_dict: Dict) -> Tuple[np.ndarray, bool]:
    """提取右手速度数据"""
    if 'right' not in data_dict:
        return np.array([]), False
    
    right_data = data_dict['right']
    if not isinstance(right_data, dict):
        return np.array([]), False
    
    transl = right_data.get('transl_worldspace', None)
    if transl is None or len(transl) == 0:
        return np.array([]), False
    
    transl = np.array(transl)
    T = len(transl)
    
    speed = np.zeros(T)
    for i in range(1, T):
        dx = transl[i, 0] - transl[i-1, 0]
        dy = transl[i, 1] - transl[i-1, 1]
        dz = transl[i, 2] - transl[i-1, 2]
        speed[i] = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return speed, True


def parse_instruction(instruction: str) -> Dict[str, str]:
    """
    解析原始指令，提取各部���
    例如: "pick the hard box and place it onto the white clock"
    返回: {'pick': "pick the hard box", 'place': "place it onto the white clock"}
    """
    result = {}
    
    # 尝试按 "and" 分割
    if ' and ' in instruction.lower():
        parts = instruction.split(' and ', 1)
        # 第一部分通常是 pick/grasp
        result['first'] = parts[0].strip()
        if len(parts) > 1:
            result['second'] = parts[1].strip()
    
    return result


def generate_instruction_from_original(original_instruction: str, minima_count: int, 
                                       segment_idx: int, total_segments: int) -> str:
    """
    根据原始指令生成分割后的指令
    
    例如:
    - 原始: "pick the hard box and place it onto the white clock"
    - segment_idx=0: "pick the hard box"
    - segment_idx=1: "place it onto the white clock"
    - segment_idx=2: "return home"

    无 ` and ` 时（仅 pick/grasp 等单句）不解析，整句抄为第一段；中间段无下半句时用 "lift it"。

    Raises:
        ValueError: 原始指令为空，或 segment_idx / minima_count 组合无法映射到指令。
    """
    text = original_instruction.strip()
    parsed = parse_instruction(original_instruction)

    if parsed:
        first = parsed.get('first', text)
        second = parsed.get('second', '')
    else:
        if not text:
            raise ValueError(
                "original_instruction is empty after strip; cannot assign per-segment instruction "
                f"(segment_idx={segment_idx}, minima_count={minima_count}, total_segments={total_segments})"
            )
        first = text
        second = ''

    if minima_count == 1:
        # 1个极小值：2段 -> 第一部分 + return home
        if segment_idx == 0:
            return first
        elif segment_idx == 1:
            return "return home"
    else:
        # 2个极小值：3段 -> 第一部分 + 第二部分 + return home
        if segment_idx == 0:
            return first
        elif segment_idx == 1:
            if second:
                # lift 指令简化为 "lift it"
                if second.lower().startswith('lift '):
                    return "lift it"
                return second
            return "lift it"
        elif segment_idx == 2:
            return "return home"

    raise ValueError(
        "no instruction mapping for this split; check caller logic. "
        f"segment_idx={segment_idx}, minima_count={minima_count}, total_segments={total_segments}, "
        f"original_instruction={original_instruction!r}"
    )


def split_episode_by_minima(npy_path: str) -> Optional[Tuple[List[Dict], List[int], str, int, str]]:
    """
    根据极小值分割episode
    
    Returns:
        (segments, minima, video_name, total_frames) 或 None if filtered out
    """
    data = np.load(npy_path, allow_pickle=True)
    if data.ndim == 0 and data.item() is not None:
        data_dict = data.item()
    else:
        data_dict = data
    
    video_name = data_dict.get('video_name', os.path.basename(npy_path).replace('.npy', ''))
    # 提取原始指令文本（从 right 字段中提取）
    text_data = data_dict.get('text', {})
    if isinstance(text_data, dict):
        right_instructions = text_data.get('right', [])
        if right_instructions and len(right_instructions) > 0:
            item = right_instructions[0]
            # item 可能是 ('pick ...', (0, 80)) 格式的元组
            if isinstance(item, tuple) and len(item) > 0:
                original_instruction = str(item[0])
            elif isinstance(item, list) and len(item) > 0:
                original_instruction = str(item[0])
            else:
                original_instruction = str(item)
        else:
            original_instruction = ''
    else:
        original_instruction = str(text_data) if text_data else ''
    
    speed, success = extract_right_hand_speed(data_dict)
    if not success or len(speed) == 0:
        return None
    
    minima = find_speed_minima(speed)
    
    # 排除极小值=0的eps
    if len(minima) == 0:
        return None
    
    # 多于2个极小值：只保留第一个和最后一个
    if len(minima) > 2:
        minima = [minima[0], minima[-1]]
    
    minima_list = minima.tolist() if hasattr(minima, 'tolist') else list(minima)
    
    T = len(speed)
    segments = []
    
    if len(minima) == 1:
        m = minima[0]
        segments.append({'segment_idx': 0, 'frame_start': 0, 'frame_end': m})
        segments.append({'segment_idx': 1, 'frame_start': m + 1, 'frame_end': T - 1})
    else:
        m1, m2 = minima[0], minima[1]
        segments.append({'segment_idx': 0, 'frame_start': 0, 'frame_end': m1})
        segments.append({'segment_idx': 1, 'frame_start': m1 + 1, 'frame_end': m2})
        segments.append({'segment_idx': 2, 'frame_start': m2 + 1, 'frame_end': T - 1})
    
    return segments, minima_list, video_name, T, original_instruction


def extract_episode_data(data_dict: Dict, frame_start: int, frame_end: int,
                         seg_idx: int, total_segments: int,
                         original_instruction: str = "") -> Dict:
    """从原始数据中提取指定帧范围的数据，并更新相关帧索引和指令文本"""
    segment_data = {}
    num_frames = frame_end - frame_start + 1

    # 生成分段后的指令（根据原始指令和 segment_idx）
    split_instruction = ""
    if original_instruction:
        split_instruction = generate_instruction_from_original(
            original_instruction,
            0,  # minima_count 不影响指令生成逻辑
            seg_idx,
            total_segments
        )

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                segment_data[key] = value
            elif len(value) > frame_end:
                segment_data[key] = value[frame_start:frame_end + 1].copy()
            else:
                segment_data[key] = value.copy() if hasattr(value, 'copy') else value
        elif isinstance(value, dict):
            segment_data[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    if sub_value.ndim == 0:
                        segment_data[key][sub_key] = sub_value
                    elif len(sub_value) > frame_end:
                        segment_data[key][sub_key] = sub_value[frame_start:frame_end + 1].copy()
                    else:
                        segment_data[key][sub_key] = sub_value.copy() if hasattr(sub_value, 'copy') else sub_value
                elif isinstance(sub_value, list):
                    # 处理 kept_frames 等列表
                    if len(sub_value) > frame_end:
                        segment_data[key][sub_key] = sub_value[frame_start:frame_end + 1]
                    else:
                        segment_data[key][sub_key] = sub_value[:]
                else:
                    segment_data[key][sub_key] = sub_value
        elif isinstance(value, list):
            if len(value) > frame_end:
                segment_data[key] = value[frame_start:frame_end + 1]
            else:
                segment_data[key] = value[:]
        else:
            segment_data[key] = value

    # 更新 video_decode_frame 为新的帧序列（从 frame_start 开始）
    if 'video_decode_frame' in segment_data:
        segment_data['video_decode_frame'] = [int(f) for f in range(frame_start, frame_end + 1)]

    # 更新 text 字段：拆分指令并更新帧范围
    if 'text' in segment_data and isinstance(segment_data['text'], dict):
        for hand_key in ['left', 'right']:
            if hand_key in segment_data['text']:
                new_text_list = []
                for item in segment_data['text'][hand_key]:
                    desc = ""
                    old_start, old_end = 0, num_frames - 1

                    if isinstance(item, tuple) and len(item) == 2:
                        desc, (old_start, old_end) = item
                    elif isinstance(item, list) and len(item) >= 2:
                        desc = item[0]
                        time_range = item[1]
                        if isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
                            old_start, old_end = time_range[0], time_range[1]

                    # 将 numpy 标量转换为普通 int
                    if hasattr(old_start, 'item'):
                        old_start = int(old_start)
                    if hasattr(old_end, 'item'):
                        old_end = int(old_end)

                    # 将绝对帧索引转换为相对于新片段起始的索引
                    new_start = max(0, old_start - frame_start)
                    new_end = min(num_frames - 1, old_end - frame_start)

                    # 如果是右手指令且有拆分指令，使用拆分后的指令
                    if hand_key == 'right' and split_instruction and desc:
                        new_desc = split_instruction
                    else:
                        new_desc = desc

                    new_text_list.append((new_desc, (int(new_start), int(new_end))))
                segment_data['text'][hand_key] = new_text_list

    return segment_data


def process_annotations_dir(annotations_dir: str, verbose: bool = True, max_eps: int = 3) -> Dict:
    """
    处理一个 episodic_annotations 目录，分割并保存到 split 目录

    Args:
        annotations_dir: episodic_annotations 目录路径
        verbose: 是否打印详细信息
        max_eps: 最多处理的 episode 数量，-1 表示全部，默认为 3

    Returns:
        处理统计信息
    """
    annotations_dir = Path(annotations_dir)
    split_dir = annotations_dir.parent / f"{annotations_dir.name}_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {annotations_dir}")
        print(f"Output to: {split_dir}")
        print(f"{'='*60}")
    
    # 查找所有npy文件
    npy_files = sorted(annotations_dir.glob("*_ep_*.npy"))
    
    if not npy_files:
        if verbose:
            print(f"No annotation files found in {annotations_dir}")
        return {'total': 0, 'kept': 0, 'filtered': 0, 'segments': 0}
    
    stats = {'total': len(npy_files), 'kept': 0, 'filtered': 0, 'segments': 0}
    
    # 收集所有分段信息用于生成json
    all_segments_info = []
    
    processed_count = 0
    for npy_path in npy_files:
        # 检查是否达到最大处理数量
        if max_eps != -1 and processed_count >= max_eps:
            if verbose:
                print(f"\n[INFO] Reached max_eps limit ({max_eps}), stopping.")
            break

        # 右手过滤
        is_valid, ratio, valid_frames, total_frames = check_right_hand_ratio(
            str(npy_path), RIGHT_HAND_THRESHOLD
        )
        if not is_valid:
            if verbose:
                print(f"  [FILTERED] {npy_path.name} (right hand ratio: {ratio*100:.1f}% = {valid_frames}/{total_frames})")
            stats['filtered'] += 1
            continue

        result = split_episode_by_minima(str(npy_path))

        if result is None:
            if verbose:
                print(f"  [FILTERED] {npy_path.name} (minima not 1 or 2)")
            stats['filtered'] += 1
            continue
        
        segments, minima, video_name, total_frames, original_instruction = result
        
        # 解析原始ep编号
        match = ANNOTATION_FILE_PATTERN.match(npy_path.name)
        if match:
            dataset_name = match.group(1)
            original_ep = match.group(2)
        else:
            dataset_name = npy_path.stem
            original_ep = "000000"
        
        # 加载原始数据
        original_data = np.load(str(npy_path), allow_pickle=True)
        if original_data.ndim == 0:
            original_data = original_data.item()
        
        stats['kept'] += 1
        
        for seg in segments:
            seg_idx = seg['segment_idx']
            frame_start = seg['frame_start']
            frame_end = seg['frame_end']

            # 生成新的ep编号: 直接递增（不以下划线分隔）
            new_ep = f"{int(original_ep) + seg_idx:06d}"
            new_ep_name = f"{dataset_name}_ep_{new_ep}"

            # 提取该段数据（只保留原始字段，进行帧切片，并更新指令文本）
            seg_data = extract_episode_data(original_data, frame_start, frame_end, 
                                            seg_idx, len(segments), original_instruction)

            # 保存npy文件（保持原始结构不变）
            output_path = split_dir / f"{new_ep_name}.npy"
            np.save(str(output_path), seg_data)

            # 收集分段信息到json（新字段只保存在json中）
            segment_info = {
                'segment_ep_name': new_ep_name,
                'original_instruction': original_instruction,
                'instruction': generate_instruction_from_original(
                    original_instruction=original_instruction,
                    minima_count=len(minima),
                    segment_idx=seg_idx,
                    total_segments=len(segments)
                ),
                'frame_start': int(frame_start),
                'frame_end': int(frame_end),
                'total_frames': int(frame_end - frame_start + 1),
                'segment_idx': int(seg_idx),
                'minima_count': int(len(minima)),
                'minima_positions': [int(m) for m in minima],
                'original_ep': original_ep,
                'npy_file': f"{new_ep_name}.npy"
            }
            all_segments_info.append(segment_info)

            if verbose:
                print(f"  [CREATED] {new_ep_name}.npy")
                print(f"           Instruction: '{segment_info['instruction']}'")
                actual_frames = len(seg_data.get('video_decode_frame', [])) if 'video_decode_frame' in seg_data else (frame_end - frame_start + 1)
                print(f"           Frames: {frame_start}-{frame_end} ({actual_frames} frames)")
                print(f"           Minima: {minima}")

            stats['segments'] += 1

        processed_count += 1
    
    # 保存分段信息json
    if all_segments_info:
        json_output_path = split_dir / "segments_info.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_segments_info, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        if verbose:
            print(f"\n  [SAVED] Segments info to: {json_output_path}")
    
    # 生成npz索引文件（保存到run目录）
    run_dir = split_dir.parent
    generate_segment_index_npz(split_dir, run_dir, verbose)
    
    if verbose:
        print(f"\n--- Summary ---")
        print(f"Total: {stats['total']}, Kept: {stats['kept']}, Filtered: {stats['filtered']}")
        print(f"Generated: {stats['segments']} segment files")
    
    return stats


def generate_segment_index_npz(split_dir: Path, run_dir: Path, verbose: bool = True):
    """
    为split目录生成episode_frame_index.npz索引文件
    
    Args:
        split_dir: split目录路径（存放npy文件）
        run_dir: run目录路径（存放npz文件）
        verbose: 是否打印详细信息
    """
    npy_files = sorted(split_dir.glob("*_ep_*.npy"))
    
    if not npy_files:
        if verbose:
            print("  [SKIP] No segment npy files found for npz generation")
        return
    
    episode_id_list = []
    episode_frames = {}
    
    for npy_path in npy_files:
        # split版本文件名已经是完整的episode_id: WorldModelV1_gpu0_task000000_ep_000000_ep_000000.npy
        # 直接使用文件名（去掉.npy）作为episode_id
        episode_id = npy_path.stem  # WorldModelV1_gpu0_task000000_ep_000000_ep_000000
        full_ep_name = episode_id

        if full_ep_name not in episode_id_list:
            episode_id_list.append(full_ep_name)
        
        # 获取帧数
        try:
            data = np.load(str(npy_path), allow_pickle=True)
            if data.ndim == 0 and data.item() is not None:
                data_dict = data.item()
                num_frames = data_dict.get('total_frames', len(data_dict.get('video_decode_frame', [])))
            else:
                num_frames = len(data)
        except:
            num_frames = 0
        
        episode_frames[full_ep_name] = num_frames
    
    # 构建index_frame_pair
    total_frames = sum(episode_frames.values())
    index_frame_pair = np.zeros((total_frames, 2), dtype=np.int64)
    
    current_frame = 0
    for ep_idx, ep_name in enumerate(episode_id_list):
        num_frames = episode_frames[ep_name]
        for frame_offset in range(num_frames):
            index_frame_pair[current_frame, 0] = ep_idx
            index_frame_pair[current_frame, 1] = frame_offset
            current_frame += 1
    
    index_to_episode_id = np.array(episode_id_list, dtype=object)
    
    # 保存npz到run目录
    npz_path = run_dir / "episode_frame_index.npz"
    np.savez(npz_path, index_frame_pair=index_frame_pair, index_to_episode_id=index_to_episode_id)
    
    if verbose:
        print(f"  [SAVED] Episode index to: {npz_path}")
        print(f"          Total episodes: {len(episode_id_list)}, Total frames: {total_frames}")


def scan_and_process_all(base_dir: str, verbose: bool = True, max_eps: int = 3) -> List[Dict]:
    """
    扫描base_dir下所有 episodic_annotations 目录并处理

    Args:
        base_dir: 基础目录（会扫描所有子目录）
        verbose: 是否打印详细信息
        max_eps: 最多处理的 episode 数量，-1 表示全部，默认为 3

    Returns:
        所有目录的处理统计
    """
    base_path = Path(base_dir)
    
    # 查找所有 episodic_annotations 目录
    annotations_dirs = []
    for item in base_path.rglob("*"):
        if item.is_dir() and item.name == "episodic_annotations":
            annotations_dirs.append(item)
    
    if not annotations_dirs:
        if verbose:
            print(f"No episodic_annotations directories found in {base_dir}")
        return []
    
    all_stats = []
    total_segments = 0
    total_filtered = 0
    
    for dir_path in sorted(annotations_dirs):
        stats = process_annotations_dir(str(dir_path), verbose, max_eps)
        all_stats.append({
            'dir': str(dir_path),
            'stats': stats
        })
        total_segments += stats['segments']
        total_filtered += stats['filtered']
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TOTAL: {total_segments} segments generated, {total_filtered} filtered out")
        print(f"{'='*60}")
    
    return all_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split episodes by speed minima and save as individual npy files with json info"
    )
    parser.add_argument(
        "path",
        help="Path to episodic_annotations directory or base directory containing multiple"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively process all subdirectories"
    )
    parser.add_argument(
        "--max-eps", "-m",
        type=int,
        default=-1,
        help="Maximum number of episodes to process. Use -1 for all episodes. Default: 3"
    )

    args = parser.parse_args()

    input_path = Path(args.path)

    if args.recursive:
        scan_and_process_all(str(input_path), max_eps=args.max_eps)
    elif input_path.is_dir() and input_path.name == "episodic_annotations":
        process_annotations_dir(str(input_path), max_eps=args.max_eps)
    else:
        # 可能是父目录，自动扫描
        scan_and_process_all(str(input_path), max_eps=args.max_eps)