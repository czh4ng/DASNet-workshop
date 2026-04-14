#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from typing import Optional
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None
from torchvision.transforms import functional as F
from scipy.signal import butter, sosfiltfilt
import fsspec


# =========================================================
# Basic utils
# =========================================================

def normalize(data: np.ndarray) -> np.ndarray:
    """Z-score + clip(-3,3), normalize along axis=0 for (N, T) arrays by default"""
    data = (data - np.mean(data, axis=0, keepdims=True)) / (np.std(data, axis=0, keepdims=True) + 1e-6)
    return np.clip(data, -3, 3)


def interpolate_line_segments_int(points):
    """Interpolate polyline points to integer grid points."""
    line_points = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        max_delta = max(abs(dx), abs(dy))
        if max_delta == 0:
            return points

        step_x = dx / max_delta
        step_y = dy / max_delta

        for j in range(max_delta + 1):
            line_points.append((int(start[0] + step_x * j), int(start[1] + step_y * j)))

    unique_points = list(set(line_points))
    unique_points.sort(key=lambda x: (x[0], x[1]))
    return unique_points


def gaussian_line(x_center, std_dev, amplitude, length):
    x = np.arange(length)
    gaussian = amplitude * np.exp(-(x - x_center) ** 2 / (2 * std_dev ** 2))
    return gaussian


def all_instances_in_allowed_bins(annotations, allowed_bins):
    """
    Only when signal_index.bin of ALL instances in this sample are within allowed ranges -> True.
    """
    if not allowed_bins:
        return True
    for ann in annotations:
        cat = ann.get("category_id")
        bin_idx = ann.get("signal_index", {}).get("bin")

        if cat not in allowed_bins:
            return False
        if bin_idx is None:
            return False

        lo, hi = allowed_bins[cat]
        if not (lo <= bin_idx <= hi):
            return False

    return True


def filter_instances_by_min_bin(annotations, min_keep_bins):
    """
    Delete weak instances:
    - If category in min_keep_bins and ann.signal_index.bin < threshold -> delete
    - If signal_index/bin missing -> delete for that category
    """
    if not min_keep_bins:
        return annotations

    kept = []
    for ann in annotations:
        cat = ann.get("category_id")
        if cat is None:
            continue

        if cat not in min_keep_bins:
            kept.append(ann)
            continue

        min_bin = min_keep_bins[cat]
        bin_idx = ann.get("signal_index", {}).get("bin", None)
        if bin_idx is None:
            continue
        if bin_idx >= min_bin:
            kept.append(ann)

    return kept


# =========================================================
# Preprocessing
# =========================================================

def _safe_design_sos_bandpass(dt: float, f1: float, f2: float, order: int = 4):
    nyq = 0.5 / dt
    wn1 = f1 / nyq
    wn2 = f2 / nyq
    # avoid edge issues
    wn1 = max(1e-6, min(wn1, 0.999999))
    wn2 = max(1e-6, min(wn2, 0.999999))
    if wn1 >= wn2:
        # fallback: tiny gap
        wn1 = max(1e-6, min(0.49, wn2 * 0.5))
    return butter(order, [wn1, wn2], btype="bandpass", output="sos")


def _safe_design_sos_highpass(dt: float, f: float, order: int = 4):
    nyq = 0.5 / dt
    wn = f / nyq
    wn = max(1e-6, min(wn, 0.999999))
    return butter(order, wn, btype="highpass", output="sos")


def preprocess_data_rgb(
    h5_path: str,
    channel_range=None,                 # (ch_start, ch_end) on axis=0 (channels)
    data_key: str = "data",
    data_is_strain_rate: bool = True,
    f_band=(2.0, 10.0),                 # bandpass for strain_rate
    f_high=10.0,                        # highpass for strain_rate
    storage_backend: str = "local",     # "local" or "gcs"
    gcs_key_path: Optional[str] = None,
):
    """
    Read raw from H5 (local or GCS) and return an RGB tensor-like array.

    - Expected dataset layout by default: f[data_key] with attrs["dt_s"]
    - Shape convention: raw data is (nch, nt); output rgb is (H, W, 3)
      where H = channels, W = time.
    """
    # --------------------------------------------------
    # 1. Open HDF5 file from local FS or GCS
    # --------------------------------------------------
    if storage_backend == "gcs" or (isinstance(h5_path, str) and h5_path.startswith("gs://")):
        fs = fsspec.filesystem("gcs", token=gcs_key_path)
        with fs.open(h5_path, "rb") as fp:
            with h5py.File(fp, "r") as f:
                if data_key not in f:
                    raise KeyError(f"'{data_key}' not found in {h5_path}. Keys={list(f.keys())}")
                dset = f[data_key]
                dt = float(dset.attrs.get("dt_s", None))
                if dt is None:
                    raise KeyError(f"dt_s attribute not found in {h5_path}:{data_key}")

                raw = dset[:]  # expected (nch, nt)
                raw = raw.astype("float32")
    else:
        with h5py.File(h5_path, "r") as f:
            if data_key not in f:
                raise KeyError(f"'{data_key}' not found in {h5_path}. Keys={list(f.keys())}")
            dset = f[data_key]
            dt = float(dset.attrs.get("dt_s", None))
            if dt is None:
                raise KeyError(f"dt_s attribute not found in {h5_path}:{data_key}")

            raw = dset[:]  # expected (nch, nt)
            raw = raw.astype("float32")

    if channel_range is not None:
        ch0, ch1 = channel_range
        raw = raw[ch0:ch1, :]

    # raw can be strain_rate or strain
    if data_is_strain_rate:
        strain_rate = raw
    else:
        # compute strain_rate = d/dt along time axis=1
        sr = np.diff(raw, axis=1) / dt
        sr = np.concatenate([np.zeros((sr.shape[0], 1), dtype=sr.dtype), sr], axis=1)
        strain_rate = sr

    # filters applied on strain_rate (per-channel along time)
    sos_bp = _safe_design_sos_bandpass(dt, f_band[0], f_band[1], order=4)
    sos_hp = _safe_design_sos_highpass(dt, f_high, order=4)

    sr_bp = sosfiltfilt(sos_bp, strain_rate, axis=1)
    sr_hp = sosfiltfilt(sos_hp, strain_rate, axis=1)

    # normalize each channel-map (keep your historical behavior: normalize over axis=0)
    ch0 = normalize(strain_rate.T).T
    ch1 = normalize(sr_bp.T).T
    ch2 = normalize(sr_hp.T).T

    rgb = np.zeros((ch0.shape[0], ch0.shape[1], 3), dtype=np.float32)  # (H=channels, W=time, 3)
    rgb[:, :, 0] = ch0
    rgb[:, :, 1] = ch1
    rgb[:, :, 2] = ch2
    return rgb, dt


def preprocess_from_array(
    data_nch_nt: np.ndarray,
    dt_s: float,
    data_is_strain_rate: bool = True,
    f_band=(2.0, 10.0),
    f_high=10.0,
):
    """
    Same preprocessing as preprocess_data_rgb but accepts a numpy array directly
    instead of reading from an HDF5 file. Used by the real-time pipeline.

    Args:
        data_nch_nt: (nch, nt) float32 array
        dt_s: sampling interval in seconds
        data_is_strain_rate: whether input is already strain rate
        f_band: bandpass frequency range
        f_high: highpass cutoff frequency

    Returns:
        rgb: (H=channels, W=time, 3) float32
        dt_s: passthrough of the sampling interval
    """
    raw = np.asarray(data_nch_nt, dtype=np.float32)

    if data_is_strain_rate:
        strain_rate = raw
    else:
        sr = np.diff(raw, axis=1) / dt_s
        sr = np.concatenate([np.zeros((sr.shape[0], 1), dtype=sr.dtype), sr], axis=1)
        strain_rate = sr

    sos_bp = _safe_design_sos_bandpass(dt_s, f_band[0], f_band[1], order=4)
    sos_hp = _safe_design_sos_highpass(dt_s, f_high, order=4)

    sr_bp = sosfiltfilt(sos_bp, strain_rate, axis=1)
    sr_hp = sosfiltfilt(sos_hp, strain_rate, axis=1)

    ch0 = normalize(strain_rate.T).T
    ch1 = normalize(sr_bp.T).T
    ch2 = normalize(sr_hp.T).T

    rgb = np.zeros((ch0.shape[0], ch0.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = ch0
    rgb[:, :, 1] = ch1
    rgb[:, :, 2] = ch2
    return rgb, dt_s


# =========================================================
# Training Dataset (full features)
# =========================================================

class DASTrainDataset(Dataset):
    """
    Training dataset:
    - COCO annotations
    - weak instance filtering (min_keep_bins)
    - allowed_bins gate for stacking
    - event stacking
    - noise stacking
    - gaussian line mask generation
    - attention_mask support
    - vertical flip augmentation
    - resize by fixed scale (not fixed size)
    - data is (C, W, H), H=channels-axis, W=time-axis in the intermediate RGB
    """

    def __init__(
        self,
        ann_path: str,
        root_dir: str,
        resize_scale: float = 0.5,

        # weak signal deletion
        min_keep_bins=None,

        # stacking control
        allowed_bins=None,

        # noise stacking
        synthetic_noise: bool = True,
        noise_csv: str = "/work/zhu-stor1/group/chun/standard_data/monterey_bay_noise/noise_list_lambda3.csv",
        syn_prob: float = 0.5,
        syn_factor_range=(0.5, 1.5),

        # event stacking
        enable_stack_event: bool = True,
        event_stack_prob: float = 0.3,
        event_alpha_range=(0.5, 1.5),

        # preprocessing
        data_key: str = "data",
        data_is_strain_rate: bool = True,
        channel_range=None,
        f_band=(2.0, 10.0),
        f_high=10.0,

        # augmentation
        enable_vflip: bool = True,
        vflip_prob: float = 0.2,  

        # gaussian mask params
        gaussian_std_dev_x: float = 50.0,
        gaussian_amplitude: float = 1.0,
    ):
        self.coco = COCO(ann_path)
        self.root_dir = root_dir
        self.ids = list(self.coco.imgs.keys())

        self.resize_scale = float(resize_scale)

        self.min_keep_bins = min_keep_bins or {}
        self.allowed_bins = allowed_bins or {}

        self.synthetic_noise = synthetic_noise
        self.noise_csv = noise_csv
        self.syn_prob = syn_prob
        self.syn_factor_range = syn_factor_range
        self.noise_df = pd.read_csv(self.noise_csv) if synthetic_noise else None

        self.enable_stack_event = enable_stack_event
        self.event_stack_prob = event_stack_prob
        self.event_alpha_range = event_alpha_range

        self.data_key = data_key
        self.data_is_strain_rate = data_is_strain_rate
        self.channel_range = channel_range
        self.f_band = f_band
        self.f_high = f_high

        self.enable_vflip = enable_vflip
        self.vflip_prob = vflip_prob

        self.gaussian_std_dev_x = gaussian_std_dev_x
        self.gaussian_amplitude = gaussian_amplitude

    def __len__(self):
        return len(self.ids)

    # -----------------------------
    # path mapping
    # -----------------------------
    def _imgid_to_h5_path(self, img_id: int) -> str:
        file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        rel = file_name.replace("_0.jpg", "")
        return os.path.join(self.root_dir, rel)

    def _load_rgb_from_imgid(self, img_id: int) -> np.ndarray:
        h5_path = self._imgid_to_h5_path(img_id)
        rgb, _dt = preprocess_data_rgb(
            h5_path,
            channel_range=self.channel_range,
            data_key=self.data_key,
            data_is_strain_rate=self.data_is_strain_rate,
            f_band=self.f_band,
            f_high=self.f_high,
        )
        return rgb  # (H, W, 3)

    # -----------------------------
    # partner selection for event stacking
    # -----------------------------
    def _select_partner(self, cur_index):
        N = len(self.ids)
        for _ in range(20):
            j = random.randint(0, N - 1)
            if j == cur_index:
                continue
            pid = self.ids[j]
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=pid))
            if all_instances_in_allowed_bins(anns, self.allowed_bins):
                return j
        return None

    def _maybe_stack_event(self, rgb_base, anns_base, index):
        if not self.enable_stack_event:
            return rgb_base, anns_base, False

        if not all_instances_in_allowed_bins(anns_base, self.allowed_bins):
            return rgb_base, anns_base, False

        if random.random() >= self.event_stack_prob:
            return rgb_base, anns_base, False

        partner_idx = self._select_partner(index)
        if partner_idx is None:
            return rgb_base, anns_base, False

        partner_id = self.ids[partner_idx]
        anns_partner = self.coco.loadAnns(self.coco.getAnnIds(imgIds=partner_id))
        rgb_partner = self._load_rgb_from_imgid(partner_id)

        # crop to min
        H = min(rgb_base.shape[0], rgb_partner.shape[0])
        W = min(rgb_base.shape[1], rgb_partner.shape[1])
        rgb_base = rgb_base[:H, :W]
        rgb_partner = rgb_partner[:H, :W]

        alpha = random.uniform(*self.event_alpha_range)
        rgb_mix = rgb_base + alpha * rgb_partner

        # normalize per channel
        for c in range(3):
            rgb_mix[:, :, c] = normalize(rgb_mix[:, :, c])

        anns_new = anns_base + anns_partner
        return rgb_mix, anns_new, True

    def _maybe_add_noise(self, rgb_input, annotations):
        """
        Only stack noise when:
        - synthetic_noise enabled
        - allowed_bins satisfied for this sample
        - pass syn_prob
        Strict shape match.
        """
        if not self.synthetic_noise:
            return rgb_input

        if not all_instances_in_allowed_bins(annotations, self.allowed_bins):
            return rgb_input

        if np.random.rand() >= self.syn_prob:
            return rgb_input

        idx = random.randint(0, len(self.noise_df) - 1)
        selected_path = self.noise_df.iloc[idx]["File Path"]

        noise_rgb, _dt = preprocess_data_rgb(
            selected_path,
            channel_range=self.channel_range,
            data_key=self.data_key,
            data_is_strain_rate=self.data_is_strain_rate,
            f_band=self.f_band,
            f_high=self.f_high,
        )

        if noise_rgb.shape != rgb_input.shape:
            # keep strict to avoid label/crop surprises
            return rgb_input

        syn_factor = random.uniform(*self.syn_factor_range)
        rgb_aug = rgb_input + noise_rgb * syn_factor

        for c in range(3):
            rgb_aug[:, :, c] = normalize(rgb_aug[:, :, c])

        return rgb_aug

    # -----------------------------
    # attention mask
    # -----------------------------
    def _generate_attention_mask(self, attention_mask_rects, mask_shape, original_shape):
        """
        attention_mask_rects: list of [x_min(time), y_min(channel), width(time), height(channel)]
        mask_shape: (new_t, new_ch) after resize  -> (time, channel)
        original_shape: (orig_t, orig_ch) before resize
        """
        attention_mask = np.zeros(mask_shape, dtype=np.uint8)

        orig_t, orig_ch = original_shape
        new_t, new_ch = mask_shape
        scale_t = new_t / max(orig_t, 1)
        scale_ch = new_ch / max(orig_ch, 1)

        for rect in attention_mask_rects:
            x_min_t, y_min_ch, width_t, height_ch = rect

            t0 = int(x_min_t * scale_t)
            ch0 = int(y_min_ch * scale_ch)
            t1 = int((x_min_t + width_t) * scale_t)
            ch1 = int((y_min_ch + height_ch) * scale_ch)

            t0 = max(0, t0)
            ch0 = max(0, ch0)
            t1 = min(new_t, t1)
            ch1 = min(new_ch, ch1)

            if t1 > t0 and ch1 > ch0:
                attention_mask[t0:t1, ch0:ch1] = 1

        return attention_mask

    # -----------------------------
    # main getitem
    # -----------------------------
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # ===== weak signal filtering =====
        annotations = filter_instances_by_min_bin(annotations, self.min_keep_bins)

        # if no instances, resample
        if len(annotations) == 0:
            return self.__getitem__(random.randint(0, len(self.ids) - 1))

        # ===== load raw -> preprocess rgb =====
        rgb_input = self._load_rgb_from_imgid(img_id)  # (H, W, 3) where H=channels, W=time

        # event stacking
        rgb_input, annotations, did_stack = self._maybe_stack_event(rgb_input, annotations, index)

        # noise stacking (only if not stacked event)
        if not did_stack:
            rgb_input = self._maybe_add_noise(rgb_input, annotations)

        # ===== convert to torch tensor =====
        # rgb_input: (H, W, 3) -> torch -> (3, W, H)
        image = torch.from_numpy(rgb_input).permute(2, 1, 0)

        # image.shape = (C, W, H)
        orig_W = image.shape[1]
        orig_H = image.shape[2]
        original_size_for_labels = (orig_W, orig_H)  # (orig_h, orig_w) in mask generation space

        # ===== resize by fixed scale =====
        new_W = max(1, int(orig_W * self.resize_scale))  # time 方向
        new_H = max(1, int(orig_H * self.resize_scale))  # channel 方向
        image = F.resize(image, (new_W, new_H))

        # COCO: x = time, y = channel
        # 模型: y = time, x = channel
        scale_t = new_W / max(orig_W, 1)   # time 缩放
        scale_ch = new_H / max(orig_H, 1)  # channel 缩放

        # mask / attention 使用的尺寸 (row=time, col=channel)
        new_h, new_w = new_W, new_H

        # ===== build targets =====
        boxes = []
        labels = []
        masks = []
        attention_masks = []
        areas = []
        iscrowd = []

        for ann in annotations:
            bbox = ann["bbox"]  # [x(time), y(channel), w(time), h(channel)]
            x0, y0, bw, bh = bbox

            # project: x=channel, y=time
            x1 = y0 * scale_ch
            y1 = x0 * scale_t
            x2 = (y0 + bh) * scale_ch
            y2 = (x0 + bw) * scale_t

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])

            # area（time * channel）
            areas.append((bw * scale_t) * (bh * scale_ch))
            iscrowd.append(ann.get("iscrowd", 0))

            # ===== gaussian line mask =====
            mask = np.zeros((new_h, new_w), dtype=np.float32)  # (time, channel)

            std_dev = self.gaussian_std_dev_x * scale_t
            amplitude = float(self.gaussian_amplitude)

            segs = ann.get("segmentation", [])
            for seg in segs:
                if len(seg) < 10:
                    continue

                pts = []
                for i in range(0, len(seg), 2):
                    t = seg[i]
                    ch = seg[i + 1]
                    x = int(ch * scale_ch)
                    y = int(t * scale_t)
                    pts.append((x, y))

                pts = interpolate_line_segments_int(pts)
                for x, y in pts:
                    if 0 <= x < new_w:
                        # Keep only the strongest response per column to avoid
                        # multi-peak accumulation around turning points.
                        mask[:, x] = np.maximum(mask[:, x], gaussian_line(y, std_dev, amplitude, new_h))

            col_max = np.max(mask, axis=0) if mask.size else np.zeros((new_w,), dtype=np.float32)
            cols = col_max > amplitude
            if np.any(cols):
                scales = col_max[cols] / amplitude
                mask[:, cols] /= scales[None, :]

            mask_u8 = (mask * 255).astype(np.uint8)
            masks.append(mask_u8)

            # ===== attention_mask support =====
            if "attention_mask" in ann:
                att = self._generate_attention_mask(
                    ann["attention_mask"],
                    mask_shape=(new_h, new_w),
                    original_shape=original_size_for_labels,  # (orig_W, orig_H)
                )
            else:
                att = np.zeros((new_h, new_w), dtype=np.uint8)

            attention_masks.append(att)

        # to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack([torch.tensor(m, dtype=torch.uint8) for m in masks])
        attention_masks = torch.stack([torch.tensor(a, dtype=torch.uint8) for a in attention_masks])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # ===== vertical flip augmentation =====
        if self.enable_vflip and (random.random() < self.vflip_prob):
            # Flip image along "height" dimension (dim=1)
            image = F.vflip(image)

            # Flip boxes in y (channel axis): y' = H - y
            H_img = image.shape[1]
            boxes[:, [1, 3]] = H_img - boxes[:, [3, 1]]

            # Flip masks and attention_masks along dim=1
            # masks shape: (N, new_h, new_w) where new_h corresponds to channel axis -> flip along dim=1
            masks = torch.flip(masks, dims=[1])
            attention_masks = torch.flip(attention_masks, dims=[1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "attention_masks": attention_masks,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd,
        }
        return image, target


# =========================================================
# Prediction Dataset
# =========================================================

class DASInferDataset(Dataset):
    """
    Prediction dataset.
    """

    def __init__(
        self,
        hdf5_files,
        resize_scale: float = 0.5,
        data_key: str = "data",
        data_is_strain_rate: bool = True,
        channel_range=None,
        f_band=(2.0, 10.0),
        f_high=10.0,
        storage_backend: str = "auto",   # "auto", "local", "gcs"
        gcs_key_path: Optional[str] = None,
    ):
        """
        Args:
            hdf5_files: list[str] of HDF5 paths (local path or gs:// URL)
            resize_scale: resize the input
            storage_backend:
                - "local": only use local file
                - "gcs":   only use GCS filesystem
                - "auto":  decide automatically
            gcs_key_path: GCS key path (json)
        """
        self.hdf5_files = list(hdf5_files)
        self.resize_scale = float(resize_scale)
        self.data_key = data_key
        self.data_is_strain_rate = data_is_strain_rate
        self.channel_range = channel_range
        self.f_band = f_band
        self.f_high = f_high
        self.storage_backend = storage_backend
        self.gcs_key_path = gcs_key_path

    def __len__(self):
        return len(self.hdf5_files)

    def _resolve_backend(self, path: str) -> str:
        if self.storage_backend == "local":
            return "local"
        if self.storage_backend == "gcs":
            return "gcs"
        if isinstance(path, str) and path.startswith("gs://"):
            return "gcs"
        return "local"

    def __getitem__(self, idx):
        file_path = self.hdf5_files[idx]
        backend = self._resolve_backend(file_path)

        rgb, _dt = preprocess_data_rgb(
            file_path,
            channel_range=self.channel_range,
            data_key=self.data_key,
            data_is_strain_rate=self.data_is_strain_rate,
            f_band=self.f_band,
            f_high=self.f_high,
            storage_backend=backend,
            gcs_key_path=self.gcs_key_path,
        )  # (H, W, 3)

        image = torch.from_numpy(rgb).permute(2, 1, 0)  # (C, W, H)

        orig_W = image.shape[1]
        orig_H = image.shape[2]
        new_W = max(1, int(orig_W * self.resize_scale))
        new_H = max(1, int(orig_H * self.resize_scale))
        image = F.resize(image, (new_W, new_H))

        return image, os.path.basename(file_path)