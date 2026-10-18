# -*- coding: utf-8 -*-
"""
============================================================================
 视频卡车扫描工具（离线分析，不影响录制）
============================================================================
 功能：
  - 用 YOLO 官方预训练模型（COCO 数据集）扫描录制好的监控长视频；
  - COCO 类别 ID 7 = truck（卡车），检出即裁剪卡车区域保存为图片；
  - 以"快进"方式扫描：每秒只取 1 帧（可配置），大幅提速；
  - 纯只读视频文件，与正在进行的 ffmpeg 录制互不干扰。

 使用方法：
  python video_check.py                       # 扫描默认目录下所有视频
  python video_check.py 路径                  # 扫描指定视频或文件夹
============================================================================
"""

import os
import sys
import json
import time
from datetime import datetime
import cv2
from ultralytics import YOLO

# ======================= 配置区域 =======================
# 默认扫描目录（录制输出根目录，会递归扫描其下所有子文件夹视频）
DEFAULT_SCAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")   # ← 替换为你的视频文件夹

# 裁剪结果保存目录
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truck_crops")   # 卡车裁剪图输出目录

# COCO 预训练模型（首次运行自动下载到脚本目录）
MODEL_PATH = "yolo11n.pt"

# COCO 类别 ID：7 = truck（卡车）
TRUCK_CLASS_ID = 7

# 检测置信度阈值
CONF_THRESHOLD = 0.5

# 快进采样：每 N 秒取 1 帧（1 = 每秒 1 帧）
SAMPLE_INTERVAL_SEC = 1

# 裁剪框外扩比例（让卡车主体带一点周边环境，便于挑选）
CROP_MARGIN = 0.1

# 裁剪图最小尺寸（宽或高低于此像素的跳过，过滤远处误检小框）
MIN_CROP_SIZE = 80

# 同一视频内连续保存图片的最小间隔（秒，视频内时间）：
# 卡车在画面中会停留多秒，避免同一辆车被连拍几十张
SAME_TRUCK_SAVE_INTERVAL = 5.0

# 扫描模式：
#   False = 跳过已扫描过的视频（通过 输出目录/.scanned.json 记录），增量扫描
#   True  = 忽略记录强制全扫
FORCE_RESCAN = False
# ===========================================================

SCAN_RECORD = os.path.join(SAVE_DIR, ".scanned.json")

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".ts", ".flv"}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_videos(paths: list) -> list:
    """收集待扫描的视频文件列表（支持传文件或文件夹，文件夹递归）"""
    videos = []
    for p in paths:
        if os.path.isfile(p):
            videos.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                        videos.append(os.path.join(root, f))
    # 排除长度为 0 的文件（录制中的半成品也可能出现，跳过）
    return sorted(v for v in videos if os.path.getsize(v) > 1024 * 1024)


def process_video(model: YOLO, video_path: str, save_dir: str) -> dict:
    """扫描单个视频，检出卡车并裁剪保存，返回统计信息"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"[跳过] 无法打开: {video_path}")
        return {"frames": 0, "hits": 0, "saved": 0}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * SAMPLE_INTERVAL_SEC)))   # 采样步长

    stats = {"frames": 0, "hits": 0, "saved": 0}
    last_save_time = -1e9   # 视频内时间戳（秒），用于同车去重

    log(f"[开始] {os.path.basename(video_path)}  "
        f"(fps={fps:.1f}, 总帧数={total_frames}, 每{step}帧取1帧)")

    frame_idx = 0
    while True:
        ret = cap.grab()   # grab 只解码不取图，跳帧时比 read 快
        if not ret:
            break

        if frame_idx % step == 0:
            ret, frame = cap.retrieve()
            if ret and frame is not None:
                stats["frames"] += 1
                video_time = frame_idx / fps

                # YOLO 推理（只认卡车类别）
                results = model.predict(
                    source=frame,
                    conf=CONF_THRESHOLD,
                    classes=[TRUCK_CLASS_ID],
                    verbose=False,
                )

                h, w = frame.shape[:2]
                for box in results[0].boxes:
                    stats["hits"] += 1
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])

                    # 外扩裁剪框并裁边
                    mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
                    cx1 = max(0, int(x1 - mx)); cy1 = max(0, int(y1 - my))
                    cx2 = min(w, int(x2 + mx)); cy2 = min(h, int(y2 + my))
                    if (cx2 - cx1) < MIN_CROP_SIZE or (cy2 - cy1) < MIN_CROP_SIZE:
                        continue

                    # 同一画面时间窗内去重：一辆车停留期只存几张
                    if video_time - last_save_time < SAME_TRUCK_SAVE_INTERVAL:
                        continue
                    last_save_time = video_time

                    crop = frame[cy1:cy2, cx1:cx2]
                    out_name = f"{base_name}_t{int(video_time)}s_{int(conf*100)}.jpg"
                    cv2.imwrite(os.path.join(save_dir, out_name), crop)
                    stats["saved"] += 1

        frame_idx += 1

    cap.release()
    log(f"[完成] 扫描帧数={stats['frames']}, 检出={stats['hits']}, 保存={stats['saved']}")
    return stats


def load_scan_record() -> dict:
    """读取已扫描视频记录：{视频路径: 文件大小}"""
    if os.path.isfile(SCAN_RECORD):
        try:
            with open(SCAN_RECORD, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_scan_record(record: dict) -> None:
    with open(SCAN_RECORD, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)


def main() -> None:
    # 扫描目标：命令行参数优先，否则用默认录制目录
    targets = sys.argv[1:] if len(sys.argv) > 1 else [DEFAULT_SCAN_DIR]
    videos = collect_videos(targets)

    if not videos:
        log("未找到可扫描的视频文件。")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 增量扫描：跳过已扫描过的视频（大小相同视为未变化）
    record = load_scan_record()
    pending = []
    skipped = 0
    for v in videos:
        size = os.path.getsize(v)
        if not FORCE_RESCAN and record.get(v) == size:
            skipped += 1
            continue
        # 大小不同说明是新文件或录制中还在增长，纳入扫描
        pending.append(v)

    log(f"共找到 {len(videos)} 个视频，跳过已扫描 {skipped} 个，本次扫描 {len(pending)} 个")
    if not pending:
        log("没有新增视频需要扫描。")
        return
    log(f"结果保存至: {SAVE_DIR}")

    # 加载模型（一次加载，全程复用）
    model = YOLO(MODEL_PATH)

    t0 = time.time()
    total_saved = 0
    for i, vp in enumerate(pending, 1):
        log(f"--- ({i}/{len(pending)}) ---")
        s = process_video(model, vp, SAVE_DIR)
        total_saved += s["saved"]
        # 注意：录制中尚未结束的文件（最近一个）大小还在增长，
        # 仍记入已扫描记录；等它写完后大小变化会再次触发扫描。
        record[vp] = os.path.getsize(vp)
        save_scan_record(record)

    log(f"全部完成：扫描 {len(pending)} 个视频，新增 {total_saved} 张裁剪图，"
        f"耗时 {(time.time()-t0)/60:.1f} 分钟。请到 {SAVE_DIR} 挑选典型图片。")


if __name__ == "__main__":
    main()

