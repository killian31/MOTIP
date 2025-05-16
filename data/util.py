# Copyright (c) Ruopeng Gao. All Rights Reserved.

from functools import lru_cache

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.nested_tensor import nested_tensor_from_tensor_list


def is_legal(annotation: dict):
    assert "id" in annotation, "Annotation must have 'id' field."
    assert "category" in annotation, "Annotation must have 'category' field."
    assert "bbox" in annotation, "Annotation must have 'bbox' field."
    assert "visibility" in annotation, "Annotation must have 'visibility' field."

    assert (
        len(annotation["id"])
        == len(annotation["category"])
        == len(annotation["bbox"])
        == len(annotation["visibility"])
    ), "The length of 'id', 'category', 'bbox', 'visibility' must be the same."

    # assert torch.unique(annotation["id"]).size(0) == annotation["id"].size(0), f"IDs must be unique."
    _id_unique = torch.unique(annotation["id"]).size(0) == annotation["id"].size(
        0
    )  # for PersonPath22

    # A hack implementation for DETR (300 queries):
    # TODO: to make it more general, maybe pass the number of queries as a parameter.
    leq_300 = annotation["id"].shape[0] <= 300

    # return len(annotation["id"]) > 0
    return len(annotation["id"]) > 0 and _id_unique and leq_300


def append_annotation(
    annotation: dict,
    obj_id: int,
    category: int,
    bbox: list,
    visibility: float,
    mask_path: str | None = None,
):
    annotation["id"] = torch.cat(
        [annotation["id"], torch.tensor([obj_id], dtype=torch.int64)]
    )
    annotation["category"] = torch.cat(
        [annotation["category"], torch.tensor([category], dtype=torch.int64)]
    )
    annotation["bbox"] = torch.cat(
        [annotation["bbox"], torch.tensor([bbox], dtype=torch.float32)]
    )
    annotation["visibility"] = torch.cat(
        [annotation["visibility"], torch.tensor([visibility], dtype=torch.float32)]
    )
    if "mask_paths" not in annotation:
        annotation["mask_paths"] = []
    if mask_path is not None:
        annotation["mask_paths"].append(mask_path)
    return annotation


def collate_fn(batch):
    images, annotations, metas = zip(*batch)  # (B, T, ...)
    _B = len(batch)
    _T = len(images[0])
    images_list = [clip[_] for clip in images for _ in range(len(clip))]
    size_divisibility = metas[0][0]["size_divisibility"]
    nested_tensor = nested_tensor_from_tensor_list(
        images_list, size_divisibility=size_divisibility
    )
    # Reshape the nested tensor:
    nested_tensor.tensors = einops.rearrange(
        nested_tensor.tensors, "(b t) c h w -> b t c h w", b=_B, t=_T
    )
    nested_tensor.mask = einops.rearrange(
        nested_tensor.mask, "(b t) h w -> b t h w", b=_B, t=_T
    )
    # Above is prepared for DETR.
    # Below is prepared for MOTIP, pre-padding the annotations:
    # max_N = max(
    #    annotation[0]["trajectory_id_labels"].shape[-1] for annotation in annotations
    # )
    max_N = 0
    for video in annotations:
        for frame in video:
            max_N = max(
                max_N,
                frame["trajectory_id_labels"].shape[-1],
                frame.get("masks", torch.zeros(0)).shape[0],
            )
    H, W = nested_tensor.tensors.shape[-2:]

    # Padding the ID annotations:
    for b in range(len(annotations)):
        for t in range(len(annotations[b])):
            frame = annotations[b][t]
            _G, _, _N = frame["trajectory_id_labels"].shape
            # pad masks
            masks = frame["masks"]  # bool tensor (N, h0, w0)
            if masks.numel():  # non-empty
                masks = (
                    F.interpolate(
                        masks.unsqueeze(1).float(),  # (N,1,h0,w0)
                        size=(H, W),
                        mode="nearest",
                    )
                    .squeeze(1)
                    .to(torch.bool)
                )  # back to (N, H, W)
            else:
                masks = torch.zeros((0, H, W), dtype=torch.bool)
            n = masks.shape[0]
            if n < max_N:
                pad = torch.zeros((max_N - n, H, W), dtype=torch.bool)
                masks = torch.cat([masks, pad], 0)
            frame["masks"] = masks
            if _N < max_N:  # pad all ID-related tensors
                pad_ids = -torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                pad_bool = torch.ones((_G, 1, max_N - _N), dtype=torch.bool)
                pad_int = -torch.ones((_G, 1, max_N - _N), dtype=torch.int64)

                frame["trajectory_id_labels"] = torch.cat(
                    [frame["trajectory_id_labels"], pad_ids], -1
                )
                frame["trajectory_id_masks"] = torch.cat(
                    [frame["trajectory_id_masks"], pad_bool], -1
                )
                frame["trajectory_ann_idxs"] = torch.cat(
                    [frame["trajectory_ann_idxs"], pad_int], -1
                )
                frame["trajectory_times"] = torch.cat(
                    [frame["trajectory_times"], t * pad_int], -1
                )
                frame["unknown_id_labels"] = torch.cat(
                    [frame["unknown_id_labels"], pad_ids], -1
                )
                frame["unknown_id_masks"] = torch.cat(
                    [frame["unknown_id_masks"], pad_bool], -1
                )
                frame["unknown_ann_idxs"] = torch.cat(
                    [frame["unknown_ann_idxs"], pad_int], -1
                )
                frame["unknown_times"] = torch.cat(
                    [frame["unknown_times"], t * pad_int], -1
                )
            pass

    for b in range(len(annotations)):
        for t in range(len(annotations[b])):
            m = annotations[b][t]["masks"]
            assert (
                m.shape[0] == max_N
            ), f"Unexpected padding: got {m.shape[0]} but expected {max_N}"
    return {
        "images": nested_tensor,
        "annotations": annotations,
        "metas": metas,
    }
