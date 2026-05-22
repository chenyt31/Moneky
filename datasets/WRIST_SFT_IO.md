# Wrist Video SFT — 模型输入 / 输出说明

一条 **episode**（例如 81 帧视频 + 标注）会被切成很多个 **训练样本**（滑动窗口）。

## 时间轴（单个样本）

在某一时刻 `t`（`hist_end = t`）取一个样本：

```
帧索引:  0    1    2   ...   t  | t+1  t+2  ...  t+K
         ├────────────────────┤ ├──────────────────┤
视频:    ✓    ✓    ✓   ...   ✓  |  (不输入未来视频)
手腕:    ✓    ✓    ✓   ...   ✓  |  ✓    ✓   ...   ✓
         └────── INPUT ───────┘ └──── TARGET ──────┘
              历史 (同步)              未来 K 帧手腕
```

- **输入**：从 episode 开头到 `t` 的**所有**历史视频帧 + **同一时间段**内每帧的左右手腕 3D 位置（相机坐标系，米）
- **输出 / 监督信号**：未来 **K=16** 帧（默认）的左右手腕 **绝对** 3D 位置（相机系），**不包含**未来视频

手腕每帧 2 只手 × 3 维 = 6 个数；某手缺失则为 `None`（batch 里用 `NaN` + `mask=False`）。

## 张量形状（`WristVideoSFTDataset.__getitem__`）

| 字段 | Shape | 含义 |
|------|-------|------|
| `history_frames` | `(T_hist, 3, H, W)` | 历史 RGB，默认 H=W=224 |
| `history_wrists` | `(T_hist, 2, 3)` | 历史手腕 xyz，左手 dim0、右手 dim1 |
| `history_wrist_mask` | `(T_hist, 2)` | 该帧该手是否有效 |
| `future_wrists` | `(K, 2, 3)` | 要预测的未来手腕轨迹 |
| `future_wrist_mask` | `(K, 2)` | 未来各帧各手是否有效 |

`T_hist = t - hist_start + 1`（默认 `hist_start=0`，即从第 0 帧一直累积到 `t`）。

## Batch 后（`WristVideoSFTCollator`）

| 字段 | Shape |
|------|-------|
| `history_frames` | `(B, T_max, 3, H, W)` |
| `history_wrists` | `(B, T_max, 2, 3)` |
| `history_wrist_mask` | `(B, T_max, 2)` |
| `history_len` | `(B,)` 每条样本真实历史长度 |
| `future_wrists` | `(B, K, 2, 3)` |
| `future_wrist_mask` | `(B, K, 2)` |

历史序列按**右对齐** padding（短序列贴在时间轴右侧，与最长样本对齐）。

## 手腕 3D 怎么来的

```text
wrist_cam = extrinsics[t] @ [transl_worldspace[t], 1]   # 前 3 维
```

不是 `joints_camspace[:, 0]`。

## 直观理解

> **看过去的视频 + 过去的手腕怎么动 → 预测接下来 K 帧手腕会去哪（相机系 3D）。**

模型不负责预测未来图像，只预测未来手腕轨迹。

## 可视化

```bash
python3 scripts/visualize_sft_io.py --hist_end 40 --future_k 16
```

生成图：`outputs/sft_io_viz/sample_io_diagram.png`
