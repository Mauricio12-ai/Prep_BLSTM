"""Muestra paso a paso cómo una seña de prueba atraviesa el modelo de fusión.

No entrena ni modifica archivos. Ejecutar desde la raíz del proyecto:
python codigo/modelos/cnn_bilstm/demo_muestra.py --id SUBJECT_5/Tongue/sample_1
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from modelo import Modelo

INDICE=Path('datos/indices/union_20261012_153325/indice_comun.csv')
LANDMARKS=Path('datos/procesados/landmarks/datos_procesados_20261005_160031/landmarks_32f.npz')
FEATURES=Path('datos/caracteristicas/resnet18')
MODELO=Path('resultados/fusion/split_comun_t5-14_v1-7-8_20261028_082612/mejor_modelo.pt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',required=True,help='sample_id, p. ej. SUBJECT_5/Tongue/sample_1')
    a=p.parse_args()
    with INDICE.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    posicion={r['sample_id']:i for i,r in enumerate(rows)}
    if a.id not in posicion:p.error('No existe esa muestra en el índice común')
    i=posicion[a.id];r=rows[i]
    clases=[n for _,n in sorted({int(x['class_id']):x['class_name'] for x in rows}.items())]
    print(f"1. MUESTRA\n   {a.id} | palabra real: {r['class_name']} | frames originales: {r['original_frames']}")
    if int(r['subject_id']) not in (5,14):print('   Aviso: no es un sujeto de prueba; el modelo pudo verla.')

    fila=int(r['landmark_row'])
    with np.load(LANDMARKS,allow_pickle=False) as d:
        xy=d['X_xy'][fila];obs=d['observed_mask'][fila];use=d['usable_mask'][fila]
    print(f'\n2. LANDMARKS PREPROCESADOS\n   XY: {xy.shape}  observed: {obs.shape}  usable: {use.shape}')
    print(f'   Puntos observados: {obs.mean():.1%} | usables tras interpolar: {use.mean():.1%}')
    lm=np.concatenate([(xy*use[...,None]).reshape(32,150),obs,use],axis=1).astype(np.float32)
    print(f'   Vector de entrada: {lm.shape} = 150 XY + 75 observed + 75 usable')

    with np.load(FEATURES/f'{i:05d}.npz',allow_pickle=False) as d:
        rgb=d['features']
    print(f'\n3. CARACTERÍSTICAS RGB (ResNet18 congelada)\n   {rgb.shape} = 32 frames con 512 valores cada uno')

    model=Modelo('fusion',64,.3,34)
    model.load_state_dict(torch.load(MODELO,map_location='cpu',weights_only=True))
    model.eval()
    R=torch.from_numpy(rgb)[None];L=torch.from_numpy(lm)[None]
    with torch.inference_mode():
        _,(h_r,_)=model.rgb(R);v_r=torch.cat((h_r[-2],h_r[-1]),dim=1)
        _,(h_l,_)=model.landmarks(L);v_l=torch.cat((h_l[-2],h_l[-1]),dim=1)
        z=torch.cat((v_r,v_l),dim=1)
        logits=model.classifier(z)
        prob=torch.softmax(logits,dim=1)[0]
    print(f'\n4. BiLSTM\n   Rama RGB: {tuple(v_r.shape)} (64 + 64)\n   Rama landmarks: {tuple(v_l.shape)} (64 + 64)')
    print(f'\n5. FUSIÓN Y CLASIFICACIÓN\n   Concatenación: {tuple(z.shape)} -> salida: {tuple(logits.shape)} (34 palabras)')
    print('\n6. PREDICCIÓN (las 3 palabras más probables)')
    valores,indices=prob.topk(3)
    for v,k in zip(valores.tolist(),indices.tolist()):print(f'   {clases[k]:<24}{v:6.1%}')
    pred=clases[int(indices[0])]
    print(f"\nRESULTADO: real = {r['class_name']} | predicha = {pred} | {'ACIERTO' if pred==r['class_name'] else 'ERROR'}")


if __name__=='__main__':main()
