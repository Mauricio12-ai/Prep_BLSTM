"""Carga una muestra conjunta y muestra las futuras entradas del modelo."""
import argparse
import json
import numpy as np


def cargar_entrada(path):
    with np.load(path, allow_pickle=False) as d:
        rgb = d['rgb']
        xy = d['X_xy']
        obs, use = d['observed_mask'], d['usable_mask']
        t = len(rgb)
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError('RGB debe ser uint8 (T,H,W,3)')
        if xy.shape != (t,75,2) or obs.shape != (t,75) or use.shape != (t,75):
            raise ValueError('Dimensiones landmarks/máscaras inconsistentes')
        if not np.isfinite(xy).all():
            raise ValueError('XY no finitas')
        indices, ids = d['source_indices'], d['source_frame_ids']
        length = int(d['original_length'])
        if not np.array_equal(indices, np.rint(np.linspace(0,length-1,t)).astype(np.int64)):
            raise ValueError('Muestreo temporal inesperado')
        if ids.shape != (t,) or np.any(np.diff(ids) < 0):
            raise ValueError('IDs RGB inválidos')
        # Mismo orden que BiLSTM.forward: 150 XY + 75 observados + 75 utilizables.
        features = np.concatenate([(xy * use[...,None]).reshape(t,150), obs, use], axis=1).astype(np.float32)
        # Ejes para CNN: batch, tiempo, canales, alto, ancho. Escala [0,1].
        # Un backbone preentrenado puede exigir otra normalización posterior.
        images = rgb.transpose(0,3,1,2).astype(np.float32) / 255.0
        report = dict(sample_id=str(d['sample_id']), label=int(d['y']),
            rgb_model_shape=[1,*images.shape], landmark_model_shape=[1,*features.shape],
            original_frames=length, selected_frames=t,
            source_frame_ids=ids.tolist(),
            selected_right_hand_fully_unobserved_ids=ids[~obs[:,54:75].any(axis=1)].tolist(),
            selected_interpolated_hand_points=int((use & ~obs)[:,33:75].sum()),
            finite_features=bool(np.isfinite(features).all()),
            status='Entrada comprobada; no se ejecutó entrenamiento ni reconocimiento')
    return images[None], features[None], report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--muestra', required=True)
    args = p.parse_args()
    _, _, report = cargar_entrada(args.muestra)
    print(json.dumps(report, ensure_ascii=False, indent=2))
