"""
Helpers for the Lab 2d workshop notebook (DASNet inference on Colab or local).

Import from the DASNet-workshop repo root (after ``os.chdir(WORK_DIR)``) so that
``dasnet`` and ``predict`` resolve like the CLI scripts.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from dasnet.data.das import DASInferDataset
from dasnet.model.dasnet import build_model

import predict as _predict


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dasnet_model() -> torch.nn.Module:
    """Same Mask R-CNN configuration as ``predict.py``."""
    return build_model(
        model_name="maskrcnn_resnet50_selectable_fpn",
        num_classes=1 + 10,
        pretrained=False,
        pretrained_backbone=True,
        trainable_backbone_layers=3,
        target_layer="P4",
        box_roi_pool_size=7,
        mask_roi_pool_size=(42, 42),
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        box_positive_fraction=0.25,
        max_size=6000,
        min_size=1422,
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()


def inference_autocast(device: torch.device):
    if device.type == "cpu":
        return nullcontext()
    ptdtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return torch.amp.autocast(device_type=device.type, dtype=ptdtype)


def make_infer_dataloader(
    h5_paths: Sequence[str],
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    resize_scale: float = 0.5,
    channel_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    storage_backend: str = "auto",
    gcs_key_path=None,
) -> Tuple[DataLoader, DASInferDataset]:
    dataset = DASInferDataset(
        list(h5_paths),
        resize_scale=resize_scale,
        data_key="data",
        data_is_strain_rate=True,
        channel_range=channel_range,
        f_band=(2.0, 10.0),
        f_high=10.0,
        storage_backend=storage_backend,
        gcs_key_path=gcs_key_path,
    )

    def collate_fn(batch):
        return tuple(zip(*batch))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader, dataset


def forward_raw(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Single forward pass; returns list of prediction dicts (before DAS postprocess)."""
    model.eval()
    dev_images = [img.to(device) for img in images]
    with torch.inference_mode(), inference_autocast(device):
        return model(dev_images)


def postprocess_batch(
    filenames: Sequence[str],
    raw_outputs: List[Dict[str, Any]],
    alpha: float = (1000 / 200) / 5.1,
):
    """NMS + mask geometry correction; same as ``predict.postprocess_dasnet``."""
    return _predict.postprocess_dasnet(list(filenames), raw_outputs, alpha=alpha)


def filter_by_score(output_i: Dict[str, Any], min_prob: float) -> Dict[str, Any]:
    scores = output_i["scores"]
    keep = scores > min_prob
    return {
        "boxes": output_i["boxes"][keep],
        "scores": scores[keep],
        "masks": output_i["masks"][keep],
        "labels": output_i["labels"][keep],
    }


def extract_peaks_for_instances(selected: Dict[str, Any]):
    """Per-instance arrival picks from masks (channel, time) indices in model space."""
    peak_points_list: List = []
    peak_values_list: List = []
    n = len(selected["scores"])
    for j in range(n):
        mask = selected["masks"][j]
        x_min, y_min, x_max, y_max = selected["boxes"][j]
        pts, vals = _predict.extract_peak_points(
            mask,
            (x_min, x_max),
            (y_min, y_max),
            threshold=0.5,
            is_filt=False,
            is_gauss=True,
        )
        peak_points_list.append(pts)
        peak_values_list.append(vals)
    return peak_points_list, peak_values_list


# Re-export for notebooks
label_map = _predict.label_map
save_predictions_json = _predict.save_predictions_json
plot_das_predictions = _predict.plot_das_predictions
