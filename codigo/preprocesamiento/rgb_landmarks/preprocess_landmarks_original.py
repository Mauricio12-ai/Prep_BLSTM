#!/usr/bin/env python3
"""Preprocesamiento reproducible de landmarks para el dataset C3 de LSP.

Entrada esperada:
Dataset/SUBJECT_n/clase/sample_n/landmarks/skeleton_real.txt

Salida:
- landmarks_32f.npz
- manifest.csv
- class_mapping.json
- loso_folds.json
- preprocessing_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

POSE = slice(0, 33)
LEFT_HAND = slice(33, 54)
RIGHT_HAND = slice(54, 75)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def natural_number(text: str) -> int:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else -1


def load_landmarks(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32)
    values = np.atleast_2d(values)
    if values.shape[1] != 225:
        raise ValueError(f"{path}: se esperaban 225 columnas y hay {values.shape[1]}")
    return values.reshape(-1, 75, 3)


def interpolate_short_gaps(
    coords: np.ndarray, max_gap: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpola huecos internos cortos por punto, sin rellenar bordes o huecos largos."""
    out = coords.copy()
    observed = np.isfinite(coords).all(axis=2)
    usable = observed.copy()
    frames, joints, _ = out.shape

    for joint in range(joints):
        missing = ~observed[:, joint]
        start = 0
        while start < frames:
            if not missing[start]:
                start += 1
                continue
            end = start
            while end < frames and missing[end]:
                end += 1
            gap = end - start
            bounded = start > 0 and end < frames
            if bounded and gap <= max_gap:
                left = out[start - 1, joint]
                right = out[end, joint]
                for offset, frame in enumerate(range(start, end), start=1):
                    alpha = offset / (gap + 1)
                    out[frame, joint] = (1 - alpha) * left + alpha * right
                    usable[frame, joint] = True
            start = end
    return out, observed, usable


def torso_normalize(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centra x/y en hombros y escala por distancia mediana entre hombros.

    La salida XY es la representación principal. La salida XYZ es híbrida:
    pose-z se centra en los hombros y hand-z conserva su referencia local,
    ambas escaladas por el mismo factor corporal.
    """
    out_xy = coords[:, :, :2].copy()
    shoulder_center_xy = (
        coords[:, LEFT_SHOULDER, :2] + coords[:, RIGHT_SHOULDER, :2]
    ) / 2.0
    shoulder_distance = np.linalg.norm(
        coords[:, LEFT_SHOULDER, :2] - coords[:, RIGHT_SHOULDER, :2], axis=1
    )
    valid_scale = shoulder_distance[np.isfinite(shoulder_distance) & (shoulder_distance > 1e-6)]
    scale = float(np.median(valid_scale)) if len(valid_scale) else np.nan
    if not np.isfinite(scale) or scale <= 1e-6:
        hip_distance = np.linalg.norm(
            coords[:, LEFT_HIP, :2] - coords[:, RIGHT_HIP, :2], axis=1
        )
        valid_hip = hip_distance[np.isfinite(hip_distance) & (hip_distance > 1e-6)]
        scale = float(np.median(valid_hip)) if len(valid_hip) else 1.0

    out_xy = (out_xy - shoulder_center_xy[:, None, :]) / scale

    out_xyz = np.empty_like(coords)
    out_xyz[:, :, :2] = out_xy
    shoulder_center_z = (
        coords[:, LEFT_SHOULDER, 2] + coords[:, RIGHT_SHOULDER, 2]
    ) / 2.0
    out_xyz[:, POSE, 2] = (
        coords[:, POSE, 2] - shoulder_center_z[:, None]
    ) / scale
    out_xyz[:, LEFT_HAND, 2] = coords[:, LEFT_HAND, 2] / scale
    out_xyz[:, RIGHT_HAND, 2] = coords[:, RIGHT_HAND, 2] / scale
    return out_xy, out_xyz, scale


def uniform_indices(length: int, target_frames: int) -> np.ndarray:
    if length < 1:
        raise ValueError("La secuencia no tiene frames")
    return np.rint(np.linspace(0, length - 1, target_frames)).astype(np.int64)


def quality_flag(length: int, usable: np.ndarray) -> str:
    if length < 8:
        return "VERY_SHORT"
    left_missing = 1.0 - usable[:, LEFT_HAND].mean()
    right_missing = 1.0 - usable[:, RIGHT_HAND].mean()
    if usable[:, LEFT_HAND].sum() == 0 or usable[:, RIGHT_HAND].sum() == 0:
        return "HAND_NEVER_DETECTED"
    if max(left_missing, right_missing) > 0.25:
        return "HIGH_HAND_MISSING"
    return "OK"


def build_folds(subjects: list[str]) -> list[dict]:
    complete = [s for s in subjects if s != "SUBJECT_17"]
    folds = []
    for test in subjects:
        candidates = [s for s in complete if s != test]
        if test in complete:
            position = complete.index(test)
            suggested_val = complete[(position + 1) % len(complete)]
        else:
            suggested_val = complete[0]
        train = [s for s in subjects if s not in {test, suggested_val}]
        folds.append(
            {
                "fold": len(folds) + 1,
                "test_subject": test,
                "suggested_validation_subject": suggested_val,
                "train_subjects": train,
                "note": "La validación sugerida usa un sujeto con las 34 clases; puede sustituirse por validación interna por sujetos.",
            }
        )
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-frames", type=int, default=32)
    parser.add_argument("--max-interpolation-gap", type=int, default=2)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    samples = sorted(
        root.glob("SUBJECT_*/*/sample_*/landmarks/skeleton_real.txt"),
        key=lambda p: (
            natural_number(p.parts[-5]),
            p.parts[-4].lower(),
            natural_number(p.parts[-3]),
        ),
    )
    if not samples:
        raise FileNotFoundError(f"No se encontraron skeleton_real.txt dentro de {root}")

    class_names = sorted({p.parts[-4] for p in samples})
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    subjects = sorted({p.parts[-5] for p in samples}, key=natural_number)

    xy_all, xyz_all = [], []
    observed_all, usable_all = [], []
    labels, subject_ids, lengths, sample_ids, flags, manifest = [], [], [], [], [], []

    for path in samples:
        subject, class_name, sample_name = path.parts[-5], path.parts[-4], path.parts[-3]
        coords = load_landmarks(path)
        cleaned, observed, usable = interpolate_short_gaps(
            coords, args.max_interpolation_gap
        )
        xy, xyz, scale = torso_normalize(cleaned)
        index = uniform_indices(len(coords), args.target_frames)
        flag = quality_flag(len(coords), usable)

        xy_all.append(np.nan_to_num(xy[index], nan=0.0, posinf=0.0, neginf=0.0))
        xyz_all.append(np.nan_to_num(xyz[index], nan=0.0, posinf=0.0, neginf=0.0))
        observed_all.append(observed[index])
        usable_all.append(usable[index])
        labels.append(class_to_id[class_name])
        subject_ids.append(natural_number(subject))
        lengths.append(len(coords))
        sample_id = f"{subject}/{class_name}/{sample_name}"
        sample_ids.append(sample_id)
        flags.append(flag)
        manifest.append(
            {
                "sample_id": sample_id,
                "subject": subject,
                "class_name": class_name,
                "class_id": class_to_id[class_name],
                "original_frames": len(coords),
                "target_frames": args.target_frames,
                "torso_scale": scale,
                "observed_points_pct": round(100 * observed.mean(), 4),
                "usable_points_pct": round(100 * usable.mean(), 4),
                "quality_flag": flag,
                "source_path": str(path.relative_to(root)),
            }
        )

    arrays = {
        "X_xy": np.asarray(xy_all, dtype=np.float32),
        "X_xyz_hybrid": np.asarray(xyz_all, dtype=np.float32),
        "observed_mask": np.asarray(observed_all, dtype=bool),
        "usable_mask": np.asarray(usable_all, dtype=bool),
        "y": np.asarray(labels, dtype=np.int64),
        "subject_id": np.asarray(subject_ids, dtype=np.int16),
        "original_length": np.asarray(lengths, dtype=np.int16),
        "sample_id": np.asarray(sample_ids),
        "quality_flag": np.asarray(flags),
    }
    np.savez_compressed(output / f"landmarks_{args.target_frames}f.npz", **arrays)

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    (output / "class_mapping.json").write_text(
        json.dumps(class_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "loso_folds.json").write_text(
        json.dumps(build_folds(subjects), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "samples": len(samples),
        "subjects": subjects,
        "classes": len(class_names),
        "target_frames": args.target_frames,
        "max_interpolation_gap": args.max_interpolation_gap,
        "landmark_order": {
            "pose": "0:33",
            "left_hand_anatomical": "33:54",
            "right_hand_anatomical": "54:75",
        },
        "normalization": "x/y centrados por frame en punto medio de hombros y divididos por distancia mediana entre hombros de la muestra",
        "xyz_warning": "X_xyz_hybrid usa z corporal centrada en hombros y z de manos en su referencia local; X_xy es la representación principal para la comparación inicial.",
        "temporal_sampling": "muestreo uniforme por vecino más cercano; conserva máscaras y evita inventar movimiento en ausencias largas",
        "quality_flags": dict(Counter(flags)),
    }
    (output / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
