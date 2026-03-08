import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from contextlib import nullcontext
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter
import utils
import json
from dasnet.data.das import DASInferDataset
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from dasnet.model.dasnet import build_model
import fsspec
import torch.distributed as dist
from torchvision.ops import box_iou
import time

label_map = {
    1: "Blue whale A",
    2: "Blue whale B",
    3: "Fin whale",
    4: "Others",
    5: "Blue whale D",
    6: "T wave",
    7: "Ship",
    8: "P wave",
    9: "S wave",
}
logger = logging.getLogger()

def get_files(data_list: str):
    """
    return hdf5 file list for prediction

    - if data_list is empty:
        auto dectect files from SeaFOAM das data
    - data_list is not empty:
            - it should include "path"
        return this path as list[str]
    """
    if data_list == "":
        if utils.is_main_process():
            print("Searching files from GCS ...")

        default_key_path = "./test_skypilot/x-berkeley-mbari-das-8c2333fca1b2.json"
        fs = fsspec.filesystem("gcs", token=default_key_path)

        hdf5_files: list[str] = []
        folders = fs.ls("berkeley-mbari-das/")
        for folder in folders:
            name = folder.split("/")[-1]
            if name in ["ContextData", "MBARI_cable_geom_dx10m.csv"]:
                continue
            years = fs.ls(folder)
            for year in years:
                jdays = fs.ls(year)
                for jday in jdays:
                    files = fs.ls(jday)
                    for file in files:
                        if file.endswith(".h5"):
                            hdf5_files.append(file)
        if utils.is_main_process():
            print(f"Total file number from GCS: {len(hdf5_files)}")
        return hdf5_files

    df = pd.read_csv(data_list)
    if "path" not in df.columns:
        raise KeyError(f'"path" column not found in {data_list}. Columns={list(df.columns)}')
    paths = df["path"].astype(str).tolist()
    if utils.is_main_process():
        print(f"Total file number from list: {len(paths)}")
    return paths
        

def extract_peak_points(matrix, x_range, y_range, threshold=0.5, sigma=5, is_filt=True, is_gauss=True):
    if is_gauss:
        matrix = gaussian_filter(matrix, sigma=sigma)

    points = []
    points_value = []
    for y in range(matrix.shape[0]):
        row = matrix[y]
        peak_x = np.argmax(row)
        peak_value = row[peak_x]
        if peak_value > threshold and x_range[0] < peak_x < x_range[1] and y_range[0] < y < y_range[1]:
            points.append([float(peak_x), float(y)])
            points_value.append(float(peak_value))

    if is_filt:
        lines = defaultdict(list)
        for x, y in points:
            lines[x].append(y)

        filtered_points = []
        for x, y_list in lines.items():
            y_list.sort()
            mid_index = len(y_list) // 2
            sampled_y = y_list[mid_index]
            filtered_points.append([x, sampled_y])
    else:
        filtered_points = sorted(points, key=lambda point: point[1])

    if len(filtered_points) == 0:
        filtered_points = []
    
    return filtered_points, points_value


def compute_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two binary masks (shape: [H, W])
    """
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    return intersection / union if union > 0 else torch.tensor(0.0, device=mask1.device)


def mask_agnostic_nms_single_image(
    prediction: dict,
    box_iou_thresh: float = 0.8,
    mask_iou_thresh: float = 0.8,
    mask_threshold: float = 0.5
) -> dict:
    """
    Applies class-agnostic, mask-aware NMS to a single image's prediction.
    Args:
        prediction: Dict with keys 'boxes', 'scores', 'labels', 'masks'
        box_iou_thresh: IoU threshold for box overlap
        mask_iou_thresh: IoU threshold for mask overlap
        mask_threshold: Threshold for converting soft masks to binary masks
    Returns:
        Filtered prediction dict with same format
    """
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    masks = prediction["masks"].squeeze(1)  # [N, H, W]

    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        current = idxs[0]
        keep.append(current.item())
        if idxs.numel() == 1:
            break

        current_box = boxes[current].unsqueeze(0)  # [1, 4]
        rest_boxes = boxes[idxs[1:]]                # [N-1, 4]
        box_ious = box_iou(current_box, rest_boxes).squeeze(0)  # [N-1]

        current_mask = (masks[current] >= mask_threshold)       # [H, W]
        rest_masks = masks[idxs[1:]] >= mask_threshold          # [N-1, H, W]
        mask_ious = torch.tensor([
            compute_mask_iou(current_mask, m) for m in rest_masks
        ], device=masks.device)

        remove_mask = (box_ious > box_iou_thresh) | (mask_ious > mask_iou_thresh)
        idxs = idxs[1:][~remove_mask]

    # Return filtered results
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
        "masks": masks[keep].unsqueeze(1)  # restore [N, 1, H, W]
    }


def postprocess_dasnet(filenames, output, alpha=(1000 / 200) / 5.1):
    """
    Post-process DASNet mask output to full image. Matches roi_heads: time extent is
    defined by channel (space) width: time_target = alpha * channel_width.

    Flow: (1) Crop mask to box region on full-image mask.
          (2) Resize cropped patch in time dimension to time_target = alpha * channel_width.
          (3) Paste back into box; if resized height exceeds image bottom, keep front part only.

    Convention: box [x_min, y_min, x_max, y_max] = (channel, time); mask (N, 1, H, W) with H=time, W=channel.
    """
    batch_size = len(filenames)
    results = []

    output = [mask_agnostic_nms_single_image(out) for out in output]

    for i in range(batch_size):
        result = {
            "boxes": output[i]["boxes"].detach().cpu().numpy(),
            "scores": output[i]["scores"].detach().cpu().numpy(),
            "labels": output[i]["labels"].detach().cpu().numpy(),
        }

        masks = output[i]["masks"]
        H_img, W_img = masks.shape[-2:]

        boxes = output[i]["boxes"]
        masks = output[i]["masks"]

        final_masks = torch.zeros((len(masks), H_img, W_img), device=masks.device)

        for j, (box, mask) in enumerate(zip(boxes, masks)):
            x_min, y_min, x_max, y_max = box.int()
            height = y_max - y_min
            channel_width = x_max - x_min
            if height < 1 or channel_width < 1:
                continue

            # 1) Crop box region from full-image mask
            cropped_mask = mask[:, y_min:y_max, x_min:x_max]

            # 2) Time extent from channel (space) width; do not clamp so we can "keep front" when pasting
            time_target = int(float(channel_width) * alpha)
            time_target = max(1, time_target)

            # 3) Resize cropped mask in time dimension to (time_target, channel_width)
            resized = F.interpolate(
                cropped_mask.unsqueeze(0),
                size=(time_target, channel_width),
                mode="bilinear",
                align_corners=False,
            )
            # resized: (1, 1, time_target, channel_width)

            # 4) Paste into image; if resized height exceeds image bottom, keep front part only
            paste_height = min(time_target, H_img - y_min)
            final_masks[j, y_min : y_min + paste_height, x_min:x_max] = resized[0, 0, :paste_height, :]

        result["masks"] = final_masks.cpu().numpy()
        results.append(result)

    return filenames, results


def save_predictions_json(file_name, selected_results, peak_points_list, peak_scores_list, output_dir, resize_scale=0.5):
    """
    Save predictions to one JSON per input file. Box and picks are in original (real) scale:
    model coordinates * (1 / resize_scale). Convention: x = channel, y = time.
    """
    scale_back = 1.0 / resize_scale
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    instances = []
    for j in range(len(selected_results["scores"])):
        box = selected_results["boxes"][j]
        box_real = [float(box[k]) * scale_back for k in range(4)]
        score = float(selected_results["scores"][j])
        label_id = int(selected_results["labels"][j])
        label_name = label_map.get(label_id, str(label_id))
        picks = peak_points_list[j] if j < len(peak_points_list) else []
        pick_scores = peak_scores_list[j] if j < len(peak_scores_list) else []
        picks_real = [[float(x) * scale_back, float(y) * scale_back] for x, y in picks]
        instances.append({
            "box": box_real,
            "score": score,
            "label": label_id,
            "label_name": label_name,
            "picks": picks_real,
            "pick_scores": [float(s) for s in pick_scores],
        })
    out = {
        "file_name": os.path.basename(file_name),
        "resize_scale": resize_scale,
        "instances": instances,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, base_name + ".json"), "w") as f:
        json.dump(out, f, indent=2)


def save_to_labelme_format(file_name, selected_results, peak_points_list, peak_scores_list, output_dir):
    shapes = []

    for i in range(len(selected_results["scores"])):
        box = selected_results["boxes"][i]
        score = float(selected_results["scores"][i])
        label_id = int(selected_results["labels"][i])
        label_name = label_map.get(label_id, str(label_id))
        peak_points = peak_points_list[i]
        peak_scores = peak_scores_list[i]

        # Box
        box_shape = {
            "label": "box",
            "points": [
                [float(box[0]), float(box[1])],
                [float(box[2]), float(box[3])]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
            "score": score
        }
        shapes.append(box_shape)

        # Peak points (LineStrip)
        if peak_points == []:
            peak_points = [[0, 0], [1, 1]]
            peak_scores = [0, 0]
        if label_id in [1, 5, 6, 7]:
            peak_points = [[0, 0], [1, 1]]
            peak_scores = [0, 0]
        line_shape = {
            "label": label_name,
            "points": [[float(x), float(y)] for x, y in peak_points],
            "point_scores": [x for x in peak_scores],
            "group_id": None,
            "description": "",
            "shape_type": "linestrip",
            "flags": {},
            "mask": None,
            "score": score
        }
        shapes.append(line_shape)

        # Confident areas (rectangle list)
        xmin = float(box[0])
        xmax = float(box[2])

        maskpoint = sorted(peak_points, key=lambda p: p[1])
        current_group = []
        confident_areas = []

        for j, (x, y) in enumerate(maskpoint):
            if not current_group or y <= current_group[-1][1] + 15:
                current_group.append((x, y))
            else:
                ymin = current_group[0][1]
                ymax = current_group[-1][1]
                if ymin == ymax:
                    if ymax < float(box[3]):
                        ymax += 1
                    else:
                        ymin -= 1
                if ymax - ymin > 50:
                    confident_areas.append((xmin, xmax, ymin, ymax))
                current_group = [(x, y)]

        if current_group:
            ymin = current_group[0][1]
            ymax = current_group[-1][1]
            if ymax - ymin > 50:
                confident_areas.append((xmin, xmax, ymin, ymax))

        for (xmin, xmax, ymin, ymax) in confident_areas:
            ca_shape = {
                "label": "confident_area",
                "points": [
                    [xmin, ymin],
                    [xmax, ymax]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
                "score": score
            }
            shapes.append(ca_shape)

    result = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": file_name + ".png",
        "imageData": "",
        "imageHeight": 2845,
        "imageWidth": 12000
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, file_name + ".json"), "w") as f:
        json.dump(result, f, indent=2)


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_das_predictions(input_data, predictions, save_path, score_threshold=0.8, mask_threshold=0.25):
    """
    Plot prediction boxes and masks on DAS image.

    Convention (aligned with das.py): image_data shape (W, H) = (time, channel);
    box (x_min, y_min, x_max, y_max) has x=channel, y=time.
    We display with x = time, y = channel: show image_data.T and map box to (time, channel).
    """
    fig, axes = plt.subplots(3, 1, figsize=(7 * 7 / 2 * 0.6, 3 * 7 * 7 / 8 * 0.6))

    for idx, ax in enumerate(axes):
        # input_data[idx]: (W, H) = (time, channel) from model image (C, W, H)
        image_data = np.array(input_data[idx], dtype=np.float32)
        # Display x=time, y=channel -> show (channel, time) so imshow puts time on x-axis
        display_img = image_data.T  # (H, W) = (channel, time)
        n_channel, n_time = display_img.shape

        plt.sca(ax)
        vmin = np.percentile(display_img, 10)
        vmax = np.percentile(display_img, 90)
        ax.imshow(display_img, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto", origin="lower", rasterized=True)

        boxes = predictions["boxes"]
        masks = predictions["masks"]
        labels = predictions["labels"]
        scores = predictions["scores"]

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if score <= score_threshold:
                continue
            # box: (x_min, y_min, x_max, y_max) = (channel_min, time_min, channel_max, time_max)
            x_min, y_min, x_max, y_max = box
            # Display coords: x = time (y_min..y_max), y = channel (x_min..x_max)
            rect = patches.Rectangle(
                (float(y_min), float(x_min)),
                float(y_max - y_min),
                float(x_max - x_min),
                linewidth=4,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            label_text = label_map.get(int(label), f"class {label}")
            ax.text(float(y_min) + 40, float(x_max) - 20, f"{label_text}\n{score:.2f}", fontsize=10)

            # mask: (time, channel) = (H_img, W_img); for overlay on display (channel, time) use mask.T
            mask_2d = np.squeeze(mask)
            if mask_2d.ndim != 2:
                continue
            display_mask = mask_2d.T  # (channel, time)

            alpha_mask = np.zeros_like(display_mask, dtype=np.float32)
            if label in [2, 3, 8, 9]:
                alpha_mask[display_mask > mask_threshold] = 0.5
            jet_colormap = plt.get_cmap("jet")
            rgba_mask = jet_colormap(display_mask)
            rgba_mask[..., 3] = alpha_mask
            ax.imshow(rgba_mask, aspect="auto", origin="lower")

            # extract_peak_points: mask (time, channel), returns (peak_x, peak_y) = (channel_idx, time_idx)
            peak_points, _ = extract_peak_points(
                mask_2d, [int(x_min), int(x_max)], [int(y_min), int(y_max)],
                threshold=0.5, is_filt=False, is_gauss=True
            )
            if len(peak_points) > 0 and label in [2, 3, 8, 9]:
                # display: x=time, y=channel -> scatter (time_idx, channel_idx)
                for px, py in peak_points:
                    ax.scatter(py, px, color="white", s=1)

        ax.set_xlim(0, n_time)
        ax.set_ylim(0, n_channel)
        ax.set_ylabel("Channel index")
        if idx == 2:
            ax.set_xlabel("Time (sample index)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def pred_dasnet(args, model, data_loader, pick_path, figure_path, resize_scale=0.5):
    """Run inference on DASNet dataset using Mask R-CNN. Results saved as JSON (box/picks in real scale)."""
    
    model.eval()
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)

    with torch.inference_mode():
        for images, filenames in tqdm(data_loader, desc="Predicting", total=len(data_loader)):
            a = time.time()
            images = [img.to(args.device) for img in images]
            
            with ctx:
                output = model(images)
                # print(f'inference time: {time.time()-a}')
                b = time.time()
                filenames, output = postprocess_dasnet(filenames, output)
                # print(f'postprocess time: {time.time()-b}')

            for i in range(len(filenames)):
                file_name = os.path.basename(filenames[i])
                scores = output[i]["scores"]
                
                # only save predictions with scores higher than threshold
                mask_threshold = args.min_prob
                keep_idx = scores > mask_threshold
                selected_results = {
                    "boxes": output[i]["boxes"][keep_idx],
                    "scores": scores[keep_idx],
                    "masks": output[i]["masks"][keep_idx],
                    "labels": output[i]["labels"][keep_idx],
                }

                if len(selected_results["scores"]) == 0:
                    continue

                peak_points_list = []
                peak_points_values = []
                for j in range(len(selected_results["scores"])):
                    mask = selected_results["masks"][j]
                    x_min, y_min, x_max, y_max = selected_results["boxes"][j]
                    peak_points, peak_point_values = extract_peak_points(
                        mask, (x_min, x_max), (y_min, y_max), threshold=0.5, is_filt=False, is_gauss=True
                    )
                    peak_points_list.append(peak_points)
                    peak_points_values.append(peak_point_values)

                save_predictions_json(
                    file_name, selected_results, peak_points_list, peak_points_values, pick_path, resize_scale
                )

                if args.plot_figure:
                    plot_das_predictions(
                        images[i].cpu().numpy(),
                        selected_results,
                        os.path.join(figure_path, file_name + ".jpg"),
                        args.min_prob
                    )

    if args.distributed:
        torch.distributed.barrier()
        dist.destroy_process_group()

    return 0


def main(args):
    result_path = args.result_path
    figure_path = os.path.join(result_path, f"figures_{args.model}")

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    utils.init_distributed_mode(args)
    print(args)

    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
    else:
        rank, world_size = 0, 1

    device = torch.device(args.device)
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    args.dtype, args.ptdtype = dtype, ptdtype
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    file_list = get_files(args.data_list)

    if args.data_list == "":
        storage_backend = "gcs"
        gcs_key_path = args.key_path
    else:
        storage_backend = "auto"
        gcs_key_path = args.key_path

    dataset = DASInferDataset(
        file_list,
        resize_scale=0.5,
        data_key="data",
        data_is_strain_rate=True,
        channel_range=None,
        f_band=(2.0, 10.0),
        f_high=10.0,
        storage_backend=storage_backend,
        gcs_key_path=gcs_key_path,
    )

    sampler = torch.utils.data.DistributedSampler(dataset) if args.distributed else None

    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=min(args.workers, mp.cpu_count()),
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = build_model( # eqnet.models.__dict__[args.model].build_model(
        model_name="maskrcnn_resnet50_selectable_fpn",
        num_classes=1+10,
        pretrained=False,
        pretrained_backbone=True,
        trainable_backbone_layers=3,
        target_layer='P4',
        box_roi_pool_size=7,
        mask_roi_pool_size=(42, 42),
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        box_positive_fraction=0.25,
        max_size=6000,
        min_size=1422
    )
    logger.info("Model:\n{}".format(model))

    model.to(device)

    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        raise ("Missing pretrained model for this location")
        # print(f"Loading pretrained model from: {args.model_path}")
        # model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.gpu])

    model.eval()
    pred_dasnet(args, model, data_loader, result_path, figure_path, resize_scale=dataset.resize_scale)

import argparse
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="DASNet Mask R-CNN Inference", add_help=add_help)

    # **Model-related parameters**
    parser.add_argument("--model", default="dasnet", type=str, help="Model name")
    # parser.add_argument("--model_path", type=str, default="", help="Path to the pre-trained model (.pth file)")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint for resuming inference")

    # **Device & computation**
    parser.add_argument("--device", default="cuda", type=str, help="Device to use: cuda / cpu")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU")
    parser.add_argument("--workers", default=4, type=int, help="Number of data loading workers")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (Automatic Mixed Precision) inference")

    # **Distributed inference**
    parser.add_argument("--distributed", action="store_true", help="Enable distributed inference")
    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="URL for setting up distributed inference")

    # **Data input**
    parser.add_argument("--data_list", type=str, default="", help="CSV file with the path of the input data")
    parser.add_argument("--object", type=str, default="default", help="TODO: Specify target files for inference (default: all)")
    parser.add_argument("--key_path", type=str, default=None, help="Path to the GCS access key (if using Google Cloud data)")

    # **Prediction settings**
    parser.add_argument("--result_path", type=str, default="results", help="Path to save inference results")
    parser.add_argument("--min_prob", default=0.5, type=float, help="Confidence threshold (predictions below this value will be discarded)")
    parser.add_argument("--plot_figure", action="store_true", help="Save prediction visualization images")

    # **DAS-specific settings**
    # parser.add_argument("--nt", default=4000, type=int, help="Number of time samples")
    # parser.add_argument("--nx", default=948, type=int, help="Number of spatial samples")
    # parser.add_argument("--cut_patch", action="store_true", help="Enable patching for continuous data")
    parser.add_argument("--skip_existing", action="store_true", help="Skip processing files that already have results")

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)