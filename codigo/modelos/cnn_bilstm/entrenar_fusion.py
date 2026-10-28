"""Entrena RGB-CNN(congelada)-BiLSTM, landmarks-BiLSTM o fusión sobre índice común."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from modelo import Modelo


def csv_write(path,fields,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def metricas(y,pred):
    cm=np.zeros((34,34),dtype=np.int64)
    np.add.at(cm,(y,pred),1)
    tp=np.diag(cm);support=cm.sum(1);counts=cm.sum(0)
    precision=np.divide(tp,counts,out=np.zeros(34),where=counts>0)
    recall=np.divide(tp,support,out=np.zeros(34),where=support>0)
    f1=np.divide(2*tp,support+counts,out=np.zeros(34),where=(support+counts)>0)
    return dict(accuracy=float(tp.sum()/cm.sum()),macro_f1_34=float(f1.mean()),
                macro_f1_supported=float(f1[support>0].mean())),cm,precision,recall,f1


def load_inputs(index,landmarks,features,mode):
    with index.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=4388 or len(set(ids))!=len(ids):
        raise ValueError('Este experimento requiere las 4388 muestras comunes, sin duplicados')
    if 'SUBJECT_2/Analyze/sample_2' in ids: raise ValueError('Se incluyó la exclusión acordada')
    with np.load(landmarks,allow_pickle=False) as d:
        ix=np.array([int(r['landmark_row']) for r in rows])
        if np.any(ix<0) or np.any(ix>=len(d['sample_id'])): raise ValueError('Fila fuera de rango')
        if d['sample_id'][ix].astype(str).tolist()!=ids: raise ValueError('Índice y landmarks distintos')
        y=d['y'][ix];subjects=d['subject_id'][ix]
        if y.tolist()!=[int(r['class_id']) for r in rows]:raise ValueError('Etiquetas distintas')
        if subjects.tolist()!=[int(r['subject_id']) for r in rows]:raise ValueError('Sujetos distintos')
        xy=d['X_xy'][ix];obs=d['observed_mask'][ix];use=d['usable_mask'][ix]
        if xy.shape!=(len(ids),32,75,2) or obs.shape!=xy.shape[:3] or use.shape!=obs.shape:
            raise ValueError('Dimensiones landmarks inesperadas')
        if not np.isfinite(xy).all():raise ValueError('XY no finitas')
        lm=np.concatenate([(xy*use[...,None]).reshape(len(ids),32,150),obs,use],axis=2).astype(np.float32)
        lengths=d['original_length'][ix]
    rgb=np.zeros((len(ids),32,1),dtype=np.float32)
    if mode!='landmarks':
        if features is None:raise ValueError('Falta --features')
        config=json.loads((features/'config_cnn.json').read_text(encoding='utf-8'))
        if config['index_sha256']!=hashlib.sha256(index.read_bytes()).hexdigest():raise ValueError('Features de otro índice')
        if config.get('weights')!='IMAGENET1K_V1' or config.get('frozen')!=True:raise ValueError('Configuración CNN inesperada')
        rgb=np.empty((len(ids),32,512),dtype=np.float32)
        for i,sid in enumerate(ids):
            with np.load(features/f'{i:05d}.npz',allow_pickle=False) as d:
                if str(d['sample_id'])!=sid:raise ValueError('Caché desordenada')
                expected=np.rint(np.linspace(0,int(lengths[i])-1,32)).astype(np.int64)
                if not np.array_equal(d['source_indices'],expected):raise ValueError('Índices temporales distintos')
                f=d['features']
                if f.shape!=(32,512) or not np.isfinite(f).all():raise ValueError('Features inválidas')
                rgb[i]=f
    classes={int(r['class_id']):r['class_name'] for r in rows}
    if set(classes)!=set(range(34)) or set(subjects.tolist())!=set(range(1,18)):
        raise ValueError('Se requieren 34 clases y 17 sujetos')
    for r in rows:
        if classes[int(r['class_id'])]!=r['class_name']:raise ValueError('Clase ambigua')
    return rows,rgb,lm,y,subjects,[classes[i] for i in range(34)]


def sujetos_de(config,name):
    """Lista de sujetos de config.json; acepta el formato antiguo de un solo sujeto."""
    value=config[name+'_subjects'] if name+'_subjects' in config else [config[name+'_subject']]
    return sorted(int(s) for s in value)


def validar_sujetos(test,val):
    if len(set(test))!=len(test) or len(set(val))!=len(val):return 'Hay sujetos repetidos en test o validación'
    if set(test)&set(val):return 'Test y validación no pueden compartir sujetos'
    if not all(1<=s<=17 for s in test) or not all(1<=s<=16 for s in val):return 'Sujetos fuera de rango; test 1–17 y val 1–16'
    if len(set(test)|set(val))>=17:return 'No quedan sujetos para entrenamiento'
    return None


def particion(subjects,test,val):
    split={'train':np.flatnonzero(~np.isin(subjects,test+val)),
           'val':np.flatnonzero(np.isin(subjects,val)),'test':np.flatnonzero(np.isin(subjects,test))}
    if any(len(v)==0 for v in split.values()):raise ValueError('Partición vacía')
    return split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indice',type=Path,required=True)
    p.add_argument('--landmarks',type=Path,required=True)
    p.add_argument('--features',type=Path)
    p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--modo',choices=['fusion','rgb','landmarks'],default='fusion')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--dropout',type=float,default=.3)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hilos',type=int,default=4)
    p.add_argument('--test-subjects','--test-subject',dest='test_subjects',type=int,nargs='+',default=[1],
                   help='Uno o varios sujetos de test, p. ej. --test-subjects 15 16')
    p.add_argument('--val-subjects','--val-subject',dest='val_subjects',type=int,nargs='+',default=[2],
                   help='Uno o varios sujetos de validación, p. ej. --val-subjects 1 5 10')
    p.add_argument('--evaluar-test',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.batch,a.hidden,a.patience,a.hilos)<1 or a.lr<=0 or not 0<=a.dropout<1:p.error('Hiperparámetros inválidos')
    error=validar_sujetos(a.test_subjects,a.val_subjects)
    if error:p.error(error)
    a.test_subjects=sorted(a.test_subjects);a.val_subjects=sorted(a.val_subjects)
    if a.salida.exists():p.error('La carpeta de salida existe; utilizar una nueva')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.hilos)
    rows,rgb,lm,y,subjects,classes=load_inputs(a.indice,a.landmarks,a.features,a.modo)
    split=particion(subjects,a.test_subjects,a.val_subjects)
    a.salida.mkdir(parents=True)
    def loader(part):
        ix=split[part]
        return DataLoader(TensorDataset(torch.from_numpy(rgb[ix]),torch.from_numpy(lm[ix]),torch.from_numpy(y[ix]).long()),
                          batch_size=a.batch,shuffle=part=='train',num_workers=0)
    train,val=loader('train'),loader('val')
    model=Modelo(a.modo,a.hidden,a.dropout)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr);criterion=nn.CrossEntropyLoss()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(device='cpu',torch=str(torch.__version__),numpy=np.__version__,
        counts={k:len(v) for k,v in split.items()},classes=classes,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        landmarks_sha256=hashlib.sha256(a.landmarks.read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256(a.indice.read_bytes()).hexdigest(),
        selection='Mejor macro F1 de validación, promedio sobre las 34 clases',
        protocol=f'Un split cross-subject (test {a.test_subjects}, val {a.val_subjects}); no LOSO completo',
        cnn='ResNet18 congelada con caché; no se entrena extremo a extremo')
    if a.modo!='landmarks':config['cnn_config']=json.loads((a.features/'config_cnn.json').read_text(encoding='utf-8'))
    (a.salida/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    (a.salida/'particiones.json').write_text(json.dumps({k:[rows[i]['sample_id'] for i in ix] for k,ix in split.items()},indent=2),encoding='utf-8')
    print('Dispositivo: CPU; modo:',a.modo,'; test:',a.test_subjects,'; val:',a.val_subjects,'; muestras:',config['counts'],flush=True)
    def evaluate(dl):
        model.eval();truth=[];pred=[];loss=0
        with torch.inference_mode():
            for r,l,t in dl:
                logits=model(r,l);loss+=criterion(logits,t).item()*len(t)
                truth.extend(t.tolist());pred.extend(logits.argmax(1).tolist())
        return loss/len(truth),metricas(np.array(truth),np.array(pred)),truth,pred
    best=-1;best_epoch=0;stale=0;history=[]
    for epoch in range(1,a.epochs+1):
        model.train();total=0;count=0
        for r,l,t in train:
            optimizer.zero_grad();loss=criterion(model(r,l),t)
            if not torch.isfinite(loss):raise ValueError('Pérdida no finita')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=loss.item()*len(t);count+=len(t)
        vl,stats,_,_=evaluate(val);m=stats[0]
        record=dict(epoch=epoch,train_loss=total/count,val_loss=vl,val_accuracy=m['accuracy'],val_macro_f1=m['macro_f1_34'])
        history.append(record);print(record,flush=True)
        csv_write(a.salida/'historial.csv',list(record),history)
        if m['macro_f1_34']>best:
            best=m['macro_f1_34'];best_epoch=epoch;stale=0
            torch.save(model.state_dict(),a.salida/'mejor_modelo.pt')
        else:stale+=1
        if stale>=a.patience:break
    model.load_state_dict(torch.load(a.salida/'mejor_modelo.pt',map_location='cpu',weights_only=True))
    report=dict(best_epoch=best_epoch,test_evaluated=False)
    for part in (['val','test'] if a.evaluar_test else ['val']):
        loss,(m,cm,precision,recall,f1),truth,pred=evaluate(loader(part))
        report[part]=dict(loss=loss,**m)
        if part=='test':report['test_evaluated']=True
        with (a.salida/f'matriz_{part}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['real / predicha',*classes]);w.writerows([[name,*cm[i]] for i,name in enumerate(classes)])
        csv_write(a.salida/f'metricas_por_clase_{part}.csv',['palabra','precision','recall','f1','support'],
            [dict(palabra=name,precision=precision[i],recall=recall[i],f1=f1[i],support=int(cm[i].sum())) for i,name in enumerate(classes)])
        csv_write(a.salida/f'predicciones_{part}.csv',['sample_id','real','predicha','correcta'],
            [dict(sample_id=rows[j]['sample_id'],real=classes[t],predicha=classes[q],correcta=t==q) for j,t,q in zip(split[part],truth,pred)])
    (a.salida/'metricas.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Finalizado. Mejor época:',best_epoch,'Test evaluado:',a.evaluar_test,flush=True)


if __name__=='__main__':main()
