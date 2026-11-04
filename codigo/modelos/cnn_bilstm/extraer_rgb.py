"""Extrae una vez ResNet18 ImageNet congelada. Reanuda al repetir el comando."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights


def read_index(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Índice vacío o IDs duplicados')
    return rows


def encode(model, images, batch):
    if images.shape != (32,112,112,3) or images.dtype != np.uint8:
        raise ValueError('RGB debe ser uint8 (32,112,112,3)')
    mean = torch.tensor([.485,.456,.406]).reshape(1,3,1,1)
    std = torch.tensor([.229,.224,.225]).reshape(1,3,1,1)
    result = []
    with torch.inference_mode():
        for start in range(0,32,batch):
            x = torch.from_numpy(images[start:start+batch].copy()).permute(0,3,1,2).float()/255
            result.append(model((x-mean)/std).cpu().numpy())
    features = np.concatenate(result).astype(np.float32)
    if features.shape != (32,512) or not np.isfinite(features).all():
        raise ValueError('Características CNN inválidas')
    return features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indice',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--limite',type=int,default=0,help='0=todas; usar 10 para prueba')
    p.add_argument('--batch-frames',type=int,default=16)
    p.add_argument('--hilos',type=int,default=4)
    a=p.parse_args()
    if a.limite<0 or min(a.batch_frames,a.hilos)<1: p.error('Argumentos numéricos inválidos')
    rows=read_index(a.indice)
    torch.set_num_threads(a.hilos)
    config=dict(version=1,index_sha256=hashlib.sha256(a.indice.read_bytes()).hexdigest(),
        samples=len(rows),backbone='resnet18',weights='IMAGENET1K_V1',frozen=True,
        spatial_input=[112,112],mean=[.485,.456,.406],std=[.229,.224,.225],
        preprocessing='RGB letterbox existente a 112; /255 y normalización ImageNet; sin crop ni resize adicional',
        torch=str(torch.__version__),torchvision=str(torchvision.__version__))
    a.salida.mkdir(parents=True,exist_ok=True)
    cp=a.salida/'config_cnn.json'
    if cp.exists():
        if json.loads(cp.read_text(encoding='utf-8'))!=config:
            raise ValueError('Cambió índice/configuración/versiones; utilizar otra carpeta de características')
    elif any(a.salida.iterdir()):
        raise ValueError('La carpeta tiene archivos sin configuración; usar una carpeta nueva')
    else: cp.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Cargando ResNet18 preentrenada; primera ejecución puede descargar pesos (~45 MB).',flush=True)
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc=nn.Identity()
    model.eval()
    model.requires_grad_(False)
    todo=rows[:a.limite] if a.limite else rows
    start=time.perf_counter(); reused=0
    for i,r in enumerate(todo):
        source=Path(r['rgb_path']); stat=source.stat()
        target=a.salida/f'{i:05d}.npz'
        if target.exists():
            with np.load(target,allow_pickle=False) as d:
                valid=(str(d['sample_id'])==r['sample_id'] and int(d['source_size'])==stat.st_size
                       and int(d['source_mtime_ns'])==stat.st_mtime_ns
                       and d['features'].shape==(32,512) and np.isfinite(d['features']).all())
            if not valid: raise ValueError('Caché distinta a entrada: '+str(target)+'; usar carpeta nueva')
            reused+=1
        else:
            with np.load(source,allow_pickle=False) as d:
                if str(d['sample_id'])!=r['sample_id']: raise ValueError('ID RGB inconsistente')
                features=encode(model,d['rgb'],a.batch_frames)
                ids=d['source_frame_ids']; indices=d['source_indices']
            temporary=target.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f,features=features,sample_id=np.array(r['sample_id']),
                    source_frame_ids=ids,source_indices=indices,
                    source_size=np.array(stat.st_size),source_mtime_ns=np.array(stat.st_mtime_ns))
            temporary.replace(target)
        if (i+1)%10==0 or i+1==len(todo):
            print(f'{i+1}/{len(todo)}; reutilizadas={reused}; segundos={time.perf_counter()-start:.1f}',flush=True)
    report=dict(requested=len(todo),index_samples=len(rows),reused=reused,
                complete=len(todo)==len(rows),seconds=time.perf_counter()-start)
    (a.salida/'resumen_extraccion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
