"""Evalúa el mejor checkpoint de entrenar_fusion.py sin entrenar ni modificarlo.

Colocar junto a modelo.py y entrenar_fusion.py. Ejecutar desde la raíz del proyecto:
python codigo/modelos/cnn_bilstm/evaluar_checkpoint.py --experimento resultados/fusion/entrenamiento_50_FECHA
Las rutas relativas de config.json se interpretan desde --raiz-proyecto (por defecto cwd).
Los resultados se guardan en una subcarpeta test_FECHA nueva; no se modifica metricas.json original.
"""
import argparse
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn
from modelo import Modelo
from entrenar_fusion import load_inputs, metricas, csv_write, sujetos_de, validar_sujetos, particion


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition,message):
    if not condition: raise ValueError(message)


def write_matrix_html(path,cm,classes):
    headers=''.join('<th class="col">'+html.escape(s)+'</th>' for s in classes)
    rows=[]
    for i,name in enumerate(classes):
        cells=[]
        total=int(cm[i].sum())
        for j,value in enumerate(cm[i]):
            value=int(value)
            alpha=.08+.75*value/max(total,1) if value else 0
            color=f'rgba(20,140,90,{alpha})' if i==j else f'rgba(210,65,65,{alpha})'
            title=html.escape(f'Real: {name}; predicha: {classes[j]}; cantidad: {value}',quote=True)
            cells.append(f'<td style="background:{color}" title="{title}">{value}</td>')
        rows.append('<tr><th class="row">'+html.escape(name)+'</th>'+''.join(cells)+f'<td>{total}</td></tr>')
    page='''<!doctype html><html lang="es"><meta charset="utf-8"><title>Matriz de confusión — test</title>
<style>body{font-family:Arial,sans-serif;margin:24px;color:#182330} .scroll{overflow:auto;max-height:80vh}
table{border-collapse:separate;border-spacing:0;font-size:12px} th,td{border:1px solid #dde2e7;padding:6px;text-align:center;min-width:29px}
th{background:#eef3f8}.row{position:sticky;left:0;text-align:left;z-index:1;min-width:170px}
.col{writing-mode:vertical-rl;transform:rotate(180deg);height:155px;white-space:nowrap}
</style><h1>Matriz de confusión de test</h1><p>Filas: palabra real. Columnas: palabra predicha.
Diagonal verde: aciertos. Fuera de la diagonal: confusiones. Cada número es un conteo de muestras.</p>
<div class="scroll"><table><thead><tr><th>Real / Predicha</th>'''+headers+ '<th>Total real</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></html>'
    path.write_text(page,encoding='utf-8')


def evaluate_experiment(experiment,root,threads=4,batch=32,overrides=None):
    experiment=Path(experiment).resolve();root=Path(root).resolve();overrides=overrides or {}
    config=json.loads((experiment/'config.json').read_text(encoding='utf-8'))
    prior=json.loads((experiment/'metricas.json').read_text(encoding='utf-8'))
    partitions=json.loads((experiment/'particiones.json').read_text(encoding='utf-8'))
    checkpoint=experiment/'mejor_modelo.pt'
    def resolve(value):
        path=Path(value)
        return path if path.is_absolute() else root/path
    # En otra PC las rutas de config.json pueden no existir; se aceptan otras, verificadas por hash.
    index=resolve(overrides.get('indice') or config['indice'])
    landmarks=resolve(overrides.get('landmarks') or config['landmarks'])
    features=overrides.get('features') or config.get('features')
    features=resolve(features) if features else None
    require(sha256(index)==config['index_sha256'],'El índice cambió desde el entrenamiento')
    require(sha256(landmarks)==config['landmarks_sha256'],'Los landmarks cambiaron desde el entrenamiento')
    if config['modo']!='landmarks':
        current=json.loads((features/'config_cnn.json').read_text(encoding='utf-8'))
        require(current==config['cnn_config'],'Configuración CNN diferente a la del entrenamiento')
    print('Cargando datos y checkpoint de la época',prior['best_epoch'],flush=True)
    rows,rgb,lm,y,subjects,classes=load_inputs(index,landmarks,features,config['modo'])
    require(classes==config['classes'],'Diccionario de clases diferente')
    test_subjects=sujetos_de(config,'test');val_subjects=sujetos_de(config,'val')
    error=validar_sujetos(test_subjects,val_subjects);require(error is None,error)
    splits=particion(subjects,test_subjects,val_subjects)
    for name,ix in splits.items():
        require([rows[i]['sample_id'] for i in ix]==partitions[name], 'Partición diferente: '+name)
        require(len(ix)==config['counts'][name],'Recuento diferente: '+name)
    test=splits['test'];require(len(test)>0,'Test vacío')
    before=sha256(checkpoint)
    torch.set_num_threads(threads)
    model=Modelo(config['modo'],config['hidden'],config['dropout'],len(classes))
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.eval();model.requires_grad_(False)
    print('Evaluando',', '.join(f'SUBJECT_{s}' for s in test_subjects)+f': {len(test)} muestras. No se actualizan pesos.',flush=True)
    criterion=nn.CrossEntropyLoss();truth=[];pred=[];loss_sum=0.0
    with torch.inference_mode():
        for start in range(0,len(test),batch):
            ix=test[start:start+batch]
            logits=model(torch.from_numpy(rgb[ix]),torch.from_numpy(lm[ix]))
            require(bool(torch.isfinite(logits).all()),'Puntuaciones no finitas')
            labels=torch.from_numpy(y[ix]).long()
            loss_sum+=criterion(logits,labels).item()*len(ix)
            truth.extend(labels.tolist());pred.extend(logits.argmax(1).tolist())
    require(sha256(checkpoint)==before,'El checkpoint cambió durante la evaluación')
    metrics,cm,precision,recall,f1=metricas(np.array(truth),np.array(pred))
    report=dict(best_epoch=int(prior['best_epoch']),test_subjects=test_subjects,val_subjects=val_subjects,
        test_evaluated=True,trained=False,samples=len(test),correct=int(np.trace(cm)),
        incorrect=int(cm.sum()-np.trace(cm)),loss=loss_sum/len(test),**metrics,
        macro_precision_34=float(precision.mean()),macro_recall_34=float(recall.mean()),
        checkpoint=str(checkpoint),checkpoint_sha256=before,
        index_sha256=config['index_sha256'],landmarks_sha256=config['landmarks_sha256'],
        protocol=f'Sujetos de test reservados {test_subjects}; no LOSO completo',torch=str(torch.__version__),
        limitation='La caché RGB se valida por IDs, formas, índices y configuración; el entrenamiento original no guardó hashes de sus características.')
    output=experiment/('test_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    (output/'metricas_test.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    csv_write(output/'predicciones_test.csv',['sample_id','y_true','y_pred','real','predicha','correcta'],
        [dict(sample_id=rows[i]['sample_id'],y_true=t,y_pred=q,real=classes[t],predicha=classes[q],correcta=t==q)
         for i,t,q in zip(test,truth,pred)])
    csv_write(output/'metricas_por_clase_test.csv',['palabra','precision','recall','f1','support'],
        [dict(palabra=name,precision=precision[i],recall=recall[i],f1=f1[i],support=int(cm[i].sum()))
         for i,name in enumerate(classes)])
    with (output/'matriz_test.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['real / predicha',*classes]);w.writerows([[name,*cm[i]] for i,name in enumerate(classes)])
    write_matrix_html(output/'matriz_test.html',cm,classes)
    print(json.dumps({k:report[k] for k in ['best_epoch','test_subjects','val_subjects','samples','correct','incorrect','accuracy','macro_f1_34','test_evaluated','trained']},indent=2))
    print('Resultados:',output)
    return output,report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experimento',type=Path,required=True,help='Carpeta que contiene mejor_modelo.pt, config.json, metricas.json y particiones.json')
    p.add_argument('--raiz-proyecto',type=Path,default=Path.cwd())
    p.add_argument('--hilos',type=int,default=4)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--indice',type=Path,help='Ruta del índice si cambió de ubicación (otra PC)')
    p.add_argument('--landmarks',type=Path,help='Ruta de landmarks_32f.npz si cambió de ubicación')
    p.add_argument('--features',type=Path,help='Carpeta de características si cambió de ubicación')
    args=p.parse_args()
    if min(args.hilos,args.batch)<1:p.error('Hilos y batch deben ser positivos')
    evaluate_experiment(args.experimento,args.raiz_proyecto,args.hilos,args.batch,
        dict(indice=args.indice,landmarks=args.landmarks,features=args.features))
