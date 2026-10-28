"""Etapa 1: pares RGB + landmarks desde ZIP; no entrena ni extrae puntos nuevos."""
import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from preprocess_landmarks_original import interpolate_short_gaps, torso_normalize, uniform_indices, quality_flag

CLASSES = ['Analyze','Better','Complaint','Crowd','Cunning','Decrease','Disappear','Envy',
 'Extra_large_cell_phone','Fail','Fine','Forget','Friday','Green','Language','Large_cell_phone',
 'Learn_everything','Learn_something','Live','People','Person','Poor','Progress','Punish','See',
 'Shout','Small','Small_cell_phone','Stagnate','Tall','Tongue','Very_poor','Very_short','Very_tall']
EXCLUDED = 'SUBJECT_2/Analyze/sample_2'


def letterbox(image, size):
    """Reduce sin deformar; rellena bordes con negro y registra transformación."""
    w, h = image.size
    ratio = min(size / w, size / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    result = np.zeros((size, size, 3), dtype=np.uint8)
    result[top:top+nh, left:left+nw] = np.asarray(image.resize((nw, nh), Image.Resampling.BILINEAR))
    mask = np.zeros((size, size), dtype=bool)
    mask[top:top+nh, left:left+nw] = True
    return result, mask, [w, h, nw, nh, left, top]


def index_zip(archive):
    """Admite Dataset/SUBJECT_n/... y SUBJECT_n/... sin extraer el ZIP."""
    groups = {}
    pattern = re.compile(r'(?:^|/)(SUBJECT_\d+)/([^/]+)/(sample_\d+)/(rgb|landmarks)/([^/]+)$')
    for info in archive.infolist():
        match = pattern.search(info.filename.replace('\\', '/'))
        if info.is_dir() or not match:
            continue
        subject, word, sample, modality, name = match.groups()
        if word not in CLASSES:
            raise ValueError('Clase desconocida: ' + word)
        key = '/'.join((subject, word, sample))
        group = groups.setdefault(key, {'rgb': {}, 'landmarks': {}})
        if modality == 'rgb':
            frame = re.fullmatch(r'rgb_frame_(\d+)\.(jpg|jpeg|png)', name, re.I)
            if not frame:
                raise ValueError('Nombre RGB no reconocido: ' + name)
            token = int(frame.group(1))
        else:
            token = name
        if token in group[modality]:
            raise ValueError('Archivo/ID duplicado: ' + info.filename)
        group[modality][token] = info
    return groups


def process_sample(archive, group, key, size, frames, gap):
    ids = np.array(sorted(group['rgb']), dtype=np.int64)
    values = np.loadtxt(io.BytesIO(archive.read(group['landmarks']['skeleton_real.txt'])),
                        dtype=np.float32, ndmin=2)
    if not len(ids) or values.shape != (len(ids), 225):
        raise ValueError(f'RGB={len(ids)}, TXT={values.shape}; se esperaba (RGB,225)')
    # Sin IDs por fila, saltos temporales requieren revisión manual.
    if np.any(np.diff(ids) != 1):
        raise ValueError('IDs RGB con saltos: revisar correspondencia con filas TXT')
    raw = values.reshape(-1, 75, 3)
    clean, observed, usable_raw = interpolate_short_gaps(raw, gap)
    xy, _, scale = torso_normalize(clean)
    # Corrección respecto al original: un centro de hombros ausente invalida XY.
    usable = usable_raw & np.isfinite(xy).all(axis=2)
    xy = np.where(usable[..., None], xy, 0).astype(np.float32)
    selected = uniform_indices(len(ids), frames)
    wanted = set(selected.tolist())
    cache = {}
    for index, fid in enumerate(ids):
        with Image.open(io.BytesIO(archive.read(group['rgb'][int(fid)]))) as im:
            im.load()  # Comprueba todas las imágenes, no solo las elegidas.
            if im.mode != 'RGB':
                raise ValueError(f'RGB con modo inesperado: {im.mode}')
            if index in wanted:
                cache[index] = letterbox(im, size)
    arrays = dict(
        rgb=np.stack([cache[i][0] for i in selected]),
        rgb_content_mask=np.stack([cache[i][1] for i in selected]),
        transforms=np.array([cache[i][2] for i in selected], dtype=np.int64),
        X_xy=xy[selected], observed_mask=observed[selected], usable_mask=usable[selected],
        source_indices=selected, source_frame_ids=ids[selected],
        original_length=np.array(len(ids)), sample_id=np.array(key),
        y=np.array(CLASSES.index(key.split('/')[1]), dtype=np.int64),
        subject_id=np.array(int(key.split('/')[0].split('_')[1])),
        quality_flag=np.array(quality_flag(len(ids), usable)),
    )
    details = dict(original_frames=len(ids), selected_frames=frames,
        unique_selected=len(set(selected.tolist())), torso_scale=scale,
        right_missing_frame_ids=ids[~observed[:, 54:75].any(axis=1)].tolist(),
        left_missing_frame_ids=ids[~observed[:, 33:54].any(axis=1)].tolist(),
        observed_hand_points=int(observed[:, 33:75].sum()),
        interpolated_hand_points=int((usable_raw & ~observed)[:, 33:75].sum()),
        points_invalidated_by_normalization=int((usable_raw & ~usable).sum()),
        quality_flag=str(arrays['quality_flag']))
    return arrays, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--salida', type=Path, required=True)
    parser.add_argument('--size', type=int, default=112)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--max-gap', type=int, default=2)
    args = parser.parse_args()
    if args.size < 16 or args.frames < 2 or args.max_gap < 0:
        parser.error('size>=16, frames>=2 y max-gap>=0')
    args.salida.mkdir(parents=True, exist_ok=False)
    config = dict(version='1.0', size=args.size, frames=args.frames, max_gap=args.max_gap,
        modalities=['RGB', 'landmarks XY'], excluded=[EXCLUDED],
        alignment='Filas TXT asociadas por orden a IDs RGB ordenados; no demuestra sincronía física.',
        changes_from_baseline='usable_mask también exige XY finitas después de normalizar.',
        quality_policy='Las banderas no reducen automáticamente el peso ni excluyen muestras.',
        source_zip=str(args.zip.resolve()))
    (args.salida/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    (args.salida/'class_mapping.json').write_text(json.dumps(dict(zip(CLASSES, range(34))), indent=2), encoding='utf-8')
    counts = {'found': 0, 'processed': 0, 'excluded': 0, 'errors': 0}
    with zipfile.ZipFile(args.zip) as archive:
        groups = index_zip(archive)
        if not groups:
            raise ValueError('No se encontraron muestras')
        counts['found'] = len(groups)
        print('Muestras encontradas:', len(groups), flush=True)
        with (args.salida/'manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id','status','output','original_frames','quality_flag','error'])
            writer.writeheader()
            for number, (key, group) in enumerate(sorted(groups.items()), 1):
                row = dict(sample_id=key, status='ERROR', output='', original_frames='', quality_flag='', error='')
                if key == EXCLUDED:
                    row.update(status='EXCLUDED', error='Exclusión común heredada del baseline; revisión pendiente')
                    counts['excluded'] += 1
                else:
                    try:
                        arrays, details = process_sample(archive, group, key, args.size, args.frames, args.max_gap)
                        relative = Path('secuencias') / (key + '.npz')
                        target = args.salida / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(target, **arrays)
                        target.with_suffix('.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
                        row.update(status='OK', output=relative.as_posix(), original_frames=details['original_frames'], quality_flag=details['quality_flag'])
                        counts['processed'] += 1
                    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as e:
                        row['error'] = str(e)
                        counts['errors'] += 1
                writer.writerow(row)
                f.flush()
                if number % 25 == 0 or number == len(groups):
                    print(f'{number}/{len(groups)}; errores={counts["errors"]}', flush=True)
    (args.salida/'resumen.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(json.dumps(counts, indent=2))
    if counts['errors']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
