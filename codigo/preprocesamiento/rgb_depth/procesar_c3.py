"""Procesa Dataset.zip sin extraerlo. Requiere numpy y Pillow."""
import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image


def letterbox(array, size, depth=False):
    h, w = array.shape[:2]
    ratio = min(size / w, size / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    im = Image.fromarray(array)
    method = Image.Resampling.NEAREST if depth else Image.Resampling.BILINEAR
    resized = np.asarray(im.resize((nw, nh), method))
    shape = (size, size) if depth else (size, size, 3)
    out = np.zeros(shape, dtype=array.dtype)
    out[top:top+nh, left:left+nw] = resized
    valid = np.zeros((size, size), dtype=bool)
    valid[top:top+nh, left:left+nw] = True
    return out, valid, [w, h, nw, nh, left, top]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', required=True, type=Path)
    p.add_argument('--salida', required=True, type=Path)
    p.add_argument('--size', type=int, default=112)
    p.add_argument('--frames', type=int, default=32)
    args = p.parse_args()
    if args.size < 16 or args.frames < 2:
        p.error('size debe ser >=16 y frames >=2')
    # Cada corrida va en carpeta nueva: protege resultados de otras configuraciones.
    args.salida.mkdir(parents=True, exist_ok=False)
    config = dict(version='1.0',size=args.size,frames=args.frames,
                  sampling='rint(linspace(0,T-1,frames))',
                  depth='uint16 original; unidades no confirmadas; no normalización métrica',
                  validation='decodificación de todas las imágenes, no solo seleccionadas',
                  alignment='IDs RGB/depth y número de filas TXT; no prueba sincronía física')
    (args.salida/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    pattern = re.compile(r'(?:^|/)Dataset/(SUBJECT_\d+)/([^/]+)/(sample_\d+)/(rgb|depth|landmarks)/([^/]+)$')
    records = []
    with zipfile.ZipFile(args.zip) as z:
        groups = {}
        for info in z.infolist():
            match = pattern.search(info.filename.replace('\\','/'))
            if not match or info.is_dir():
                continue
            subject, word, sample, modality, filename = match.groups()
            key = '/'.join((subject,word,sample))
            group = groups.setdefault(key, {'rgb':{},'depth':{},'landmarks':{}})
            if modality == 'landmarks':
                token = filename
            else:
                m = re.fullmatch(r'(?:rgb|depth)_frame_(\d+)\.(?:jpg|png)',filename,re.I)
                if not m:
                    raise ValueError('Nombre de imagen no reconocido: '+info.filename)
                token = int(m.group(1))
            if token in group[modality]:
                raise ValueError('ID duplicado: '+info.filename)
            group[modality][token] = info
        if not groups:
            raise ValueError('No se encontraron muestras con la estructura esperada.')
        print(f'Muestras encontradas: {len(groups)}',flush=True)
        fields=['sample_id','status','frames','unique_selected','frame_gaps','output','error']
        with (args.salida/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as mf:
            writer=csv.DictWriter(mf,fieldnames=fields);writer.writeheader()
            for index,(key,group) in enumerate(sorted(groups.items()),1):
                row=dict(sample_id=key,status='ERROR',frames=0,unique_selected=0,frame_gaps=0,output='',error='')
                try:
                    ids=sorted(group['rgb'])
                    if not ids or ids!=sorted(group['depth']):
                        raise ValueError('IDs RGB/depth diferentes o secuencia vacía')
                    row['frames']=len(ids)
                    row['frame_gaps']=sum(b-a!=1 for a,b in zip(ids,ids[1:]))
                    for filename,columns in [('skeleton_pixel.txt',150),('skeleton_real.txt',225)]:
                        values=np.loadtxt(io.BytesIO(z.read(group['landmarks'][filename])),ndmin=2)
                        if values.shape!=(len(ids),columns):
                            raise ValueError(f'{filename}: forma {values.shape}, esperada {(len(ids),columns)}')
                    selected=np.rint(np.linspace(0,len(ids)-1,args.frames)).astype(np.int64)
                    wanted={ids[i] for i in selected}
                    cache={}
                    for fid in ids:
                        pair=[]
                        for modality in ['rgb','depth']:
                            with Image.open(io.BytesIO(z.read(group[modality][fid]))) as im:
                                im.load()
                                if modality=='rgb':
                                    if im.mode!='RGB':
                                        raise ValueError(f'RGB inesperado: {im.mode}')
                                    a=np.asarray(im)
                                else:
                                    a=np.asarray(im)
                                    if a.ndim!=2 or a.dtype!=np.uint16:
                                        raise ValueError(f'Depth debe ser uint16 monocanal: {a.dtype}/{a.shape}')
                                if fid in wanted:
                                    pair.append(letterbox(a,args.size,modality=='depth'))
                        if fid in wanted:
                            cache[fid]=pair
                    rgb=np.stack([cache[ids[i]][0][0] for i in selected])
                    depth=np.stack([cache[ids[i]][1][0] for i in selected])
                    rgb_valid=np.stack([cache[ids[i]][0][1] for i in selected])
                    depth_valid=np.stack([cache[ids[i]][1][1] for i in selected])
                    transform=np.array([[cache[ids[i]][0][2],cache[ids[i]][1][2]] for i in selected])
                    out=args.salida/'secuencias'/Path(key+'.npz')
                    out.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(out,rgb=rgb,depth_raw=depth,
                        rgb_content_mask=rgb_valid,depth_content_mask=depth_valid,
                        depth_nonzero_mask=depth_valid & (depth!=0),
                        source_frame_ids=np.array(ids)[selected],source_indices=selected,
                        original_length=np.array(len(ids)),sample_id=np.array(key),
                        transforms=transform)
                    row.update(status='OK',unique_selected=len(wanted),output=out.relative_to(args.salida).as_posix())
                except (ValueError,KeyError,OSError,zipfile.BadZipFile) as error:
                    row['error']=str(error)
                writer.writerow(row);mf.flush();records.append(row)
                if index%25==0 or index==len(groups):
                    print(f'{index}/{len(groups)}; errores: {sum(r["status"]!="OK" for r in records)}',flush=True)
    summary={'samples':len(records),'processed':sum(r['status']=='OK' for r in records),
             'errors':sum(r['status']!='OK' for r in records),
             'short_under_8':sum(0<r['frames']<8 for r in records)}
    (args.salida/'resumen.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
