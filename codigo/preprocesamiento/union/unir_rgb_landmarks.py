"""Paso 2: comprueba y relaciona RGB preprocesado con landmarks, sin copiar datos.

Requiere numpy. No entrena. Salidas: indice_comun.csv, revision.csv y resumen.json.
El índice contiene la ruta RGB y el número de fila del NPZ de landmarks.
No modifica las entradas ni corrige puntos anatómicos. Las máscaras se conservan.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import numpy as np

EXCLUDED = 'SUBJECT_2/Analyze/sample_2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(landmarks, rgb_root, output):
    landmarks, rgb_root, output = map(lambda p: Path(p).resolve(), (landmarks, rgb_root, output))
    require(landmarks.is_file(), 'No existe el NPZ de landmarks: ' + str(landmarks))
    require((rgb_root/'manifest.csv').is_file(), 'No existe manifest.csv de RGB')
    require(not output.exists(), 'La carpeta de salida ya existe; utiliza otro nombre')
    with np.load(landmarks, allow_pickle=False) as data:
        arrays = {k: data[k] for k in ['sample_id','X_xy','observed_mask','usable_mask',
                                      'y','subject_id','original_length','quality_flag']}
    ids = arrays['sample_id'].astype(str)
    n = len(ids)
    require(n > 0 and len(set(ids)) == n, 'IDs de landmarks vacíos o duplicados')
    require(arrays['X_xy'].shape == (n,32,75,2), 'XY debe tener forma (N,32,75,2)')
    for key in ['observed_mask','usable_mask']:
        require(arrays[key].shape == (n,32,75) and arrays[key].dtype == bool, 'Máscara inválida: ' + key)
    for key in ['y','subject_id','original_length','quality_flag']:
        require(arrays[key].shape == (n,), 'Dimensión inválida: ' + key)
    with (rgb_root/'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        rgb_rows = list(csv.DictReader(f))
    rgb_by_id = {}
    for row in rgb_rows:
        require(row['sample_id'] not in rgb_by_id, 'ID RGB duplicado: ' + row['sample_id'])
        rgb_by_id[row['sample_id']] = row
    mapping_path = landmarks.parent/'class_mapping.json'
    require(mapping_path.is_file(), 'Falta class_mapping.json junto a landmarks_32f.npz')
    mapping = json.loads(mapping_path.read_text(encoding='utf-8-sig'))
    require(len(mapping) == 34 and set(mapping.values()) == set(range(34)), 'Se requieren las 34 clases originales')
    output.mkdir(parents=True)
    accepted, review = [], []
    for index, sid in enumerate(ids):
        row = dict(sample_id=sid, status='ERROR', reason='')
        if sid == EXCLUDED:
            row.update(status='EXCLUDED', reason='Exclusión común heredada; alineación pendiente de revisión')
        else:
            try:
                parts = sid.split('/')
                require(len(parts) == 3 and all(p not in ('','.','..') for p in parts), 'ID inválido')
                subject, word, _ = parts
                label = int(arrays['y'][index])
                require(word in mapping and mapping[word] == label, 'Clase no coincide con su etiqueta')
                require(subject == 'SUBJECT_' + str(int(arrays['subject_id'][index])), 'Sujeto inconsistente')
                require(1 <= int(arrays['subject_id'][index]) <= 17, 'Sujeto fuera de rango')
                require(np.isfinite(arrays['X_xy'][index]).all(), 'XY no finitas')
                require(not np.any(arrays['observed_mask'][index] & ~arrays['usable_mask'][index]),
                        'Máscaras incompatibles con el baseline original; revisar versión')
                require(sid in rgb_by_id, 'Muestra ausente del manifest RGB')
                rgb_row = rgb_by_id[sid]
                require(rgb_row['status'] == 'OK', 'RGB no procesado: ' + rgb_row.get('error',''))
                path = (rgb_root / PurePosixPath(rgb_row['output'])).resolve()
                require(rgb_root in path.parents and path.is_file(), 'Ruta RGB inexistente o fuera de carpeta')
                with np.load(path, allow_pickle=False) as rgb:
                    require(str(rgb['sample_id']) == sid, 'sample_id interno RGB diferente')
                    length = int(arrays['original_length'][index])
                    require(length > 0 and int(rgb['original_length']) == length, 'Longitudes originales diferentes')
                    expected = np.rint(np.linspace(0,length-1,32)).astype(np.int64)
                    require(np.array_equal(rgb['source_indices'], expected), 'Índices RGB distintos al muestreo landmarks esperado')
                    frame_ids = rgb['source_frame_ids']
                    require(frame_ids.shape == (32,) and np.issubdtype(frame_ids.dtype,np.integer), 'IDs RGB inválidos')
                    # Exige numeración consecutiva en los instantes guardados.
                    require(np.all(frame_ids - expected == frame_ids[0]), 'IDs con saltos/desfase: revisión manual')
                    images = rgb['rgb']
                    require(images.shape == (32,112,112,3) and images.dtype == np.uint8, 'Forma/tipo RGB inesperado')
                accepted.append(dict(sample_id=sid, landmark_row=index, rgb_path=str(path),
                    class_name=word, class_id=label, subject_id=int(arrays['subject_id'][index]),
                    original_frames=length, quality_flag=str(arrays['quality_flag'][index])))
                row.update(status='OK')
            except (ValueError, KeyError, OSError, EOFError) as e:
                row['reason'] = str(e)
        review.append(row)
        if (index+1) % 100 == 0 or index+1 == n:
            print(f'Revisadas {index+1}/{n}; compatibles: {len(accepted)}', flush=True)
    for sid in sorted(set(rgb_by_id) - set(ids)):
        review.append(dict(sample_id=sid,status='ERROR',reason='ID RGB sin fila de landmarks'))
    write_csv(output/'indice_comun.csv', accepted,
              ['sample_id','landmark_row','rgb_path','class_name','class_id','subject_id','original_frames','quality_flag'])
    write_csv(output/'revision.csv', review, ['sample_id','status','reason'])
    counts = {status: sum(r['status'] == status for r in review) for status in ['OK','EXCLUDED','ERROR']}
    by_subject = {str(s): sum(r['subject_id'] == s for r in accepted) for s in range(1,18)}
    report = dict(landmark_samples=n, rgb_manifest_samples=len(rgb_rows), compatible=counts['OK'],
        excluded=counts['EXCLUDED'], errors=counts['ERROR'], by_subject=by_subject,
        landmarks_path=str(landmarks), rgb_root=str(rgb_root),
        sample_ids_sha256=hashlib.sha256('\n'.join(r['sample_id'] for r in accepted).encode()).hexdigest(),
        temporal_check='Índices landmarks reconstruidos con rint(linspace(0,T-1,32)); el NPZ original no guarda índices.',
        limitation='Compatibilidad estructural bajo muestreo declarado; no prueba sincronía física ni exactitud de detecciones.',
        mask_policy='Se conservan las máscaras originales. XY finitas no demuestra que la normalización original fuera válida antes de reemplazar NaN.',
        next_step='Revisar errores antes de entrenar; usar el mismo índice común en todas las variantes.',
        trained=False)
    (output/'resumen.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'class_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['landmark_samples','rgb_manifest_samples','compatible','excluded','errors']},indent=2))
    print('Resultados:', output)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--landmarks', type=Path, required=True)
    p.add_argument('--rgb-root', type=Path, required=True)
    p.add_argument('--salida', type=Path, required=True)
    args = p.parse_args()
    result = run(args.landmarks, args.rgb_root, args.salida)
    if result['errors']:
        raise SystemExit(2)
