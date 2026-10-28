"""Baseline BiLSTM C3. Prueba técnica por defecto; test externo solo con --evaluar-test."""
import argparse, csv, hashlib, json, random
from pathlib import Path
from exportar_matrices import exportar
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXCLUDED = 'SUBJECT_2/Analyze/sample_2'

class BiLSTM(nn.Module):
    def __init__(self, hidden=64, dropout=0.3):
        super().__init__()
        # 150 coordenadas + 75 indicadores observados + 75 utilizables.
        self.encoder = nn.LSTM(300, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden*2,34))

    def forward(self, xy, observed, usable):
        xy = xy * usable.unsqueeze(-1)
        features = torch.cat([xy.flatten(2), observed, usable],dim=-1)
        _, (hidden, _) = self.encoder(features)
        return self.classifier(torch.cat([hidden[-2],hidden[-1]],dim=1))

def metrics(y, pred):
    cm=np.zeros((34,34),dtype=np.int64)
    np.add.at(cm,(y,pred),1)
    tp=np.diag(cm); support=cm.sum(1); predicted=cm.sum(0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(34),where=(support+predicted)>0)
    return dict(accuracy=float(tp.sum()/cm.sum()),macro_f1_34=float(f1.mean()),
                macro_f1_supported=float(f1[support>0].mean()),
                support=support.tolist(),confusion=cm.tolist())

def load_data(path):
    with np.load(path,allow_pickle=False) as d:
        ids=d['sample_id'].astype(str)
        if len(set(ids))!=len(ids): raise ValueError('sample_id duplicados')
        keep=ids!=EXCLUDED
        xy=d['X_xy'][keep]; observed=d['observed_mask'][keep]; usable=d['usable_mask'][keep]
        y=d['y'][keep]; subjects=d['subject_id'][keep]; ids=ids[keep]
    if len(ids)!=4388: raise ValueError(f'Se esperaban 4388 muestras comunes, hay {len(ids)}')
    if xy.shape!=(4388,32,75,2) or observed.shape!=xy.shape[:3] or usable.shape!=observed.shape:
        raise ValueError('Dimensiones inesperadas')
    if not np.isfinite(xy).all() or np.any(observed & ~usable): raise ValueError('Coordenadas/máscaras inválidas')
    if set(y.tolist())!=set(range(34)) or set(subjects.tolist())!=set(range(1,18)):
        raise ValueError('Clases o sujetos inesperados')
    return xy,observed,usable,y,subjects,ids

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datos',required=True,type=Path)
    p.add_argument('--salida',required=True,type=Path)
    p.add_argument('--test-subject',type=int,default=1)
    p.add_argument('--val-subject',type=int,default=2)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--evaluar-test',action='store_true')
    args=p.parse_args()
    if args.test_subject==args.val_subject or not 1<=args.test_subject<=17 or not 1<=args.val_subject<=16:
        p.error('Test y validación deben ser sujetos distintos; validación entre 1 y 16')
    if min(args.epochs,args.batch,args.hidden,args.patience)<1 or args.lr<=0:p.error('Parámetros deben ser positivos')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    xy,obs,use,y,subjects,ids=load_data(args.datos)
    split={'train':np.where((subjects!=args.test_subject)&(subjects!=args.val_subject))[0],
           'val':np.where(subjects==args.val_subject)[0], 'test':np.where(subjects==args.test_subject)[0]}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.salida.mkdir(parents=True,exist_ok=False)
    def loader(part):
        i=split[part]
        ds=TensorDataset(torch.from_numpy(xy[i]).float(),torch.from_numpy(obs[i]).float(),
                         torch.from_numpy(use[i]).float(),torch.from_numpy(y[i]).long())
        return DataLoader(ds,batch_size=args.batch,shuffle=part=='train',num_workers=0)
    train,validation=loader('train'),loader('val')
    model=BiLSTM(args.hidden).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion=nn.CrossEntropyLoss()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(device=str(device),torch_version=str(torch.__version__),numpy_version=np.__version__,
                  excluded=EXCLUDED,parameters=sum(x.numel() for x in model.parameters()),
                  counts={k:len(v) for k,v in split.items()},
                  sample_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                  status='baseline inicial; sin tuning; sin anotación de pares mínimos')
    (args.salida/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    (args.salida/'particiones.json').write_text(json.dumps({k:ids[v].tolist() for k,v in split.items()},indent=2),encoding='utf-8')
    print('Dispositivo:',device,'Muestras:',config['counts'],flush=True)
    def evaluate(dl):
        model.eval();targets=[];pred=[];loss=0;count=0
        with torch.no_grad():
            for x,o,u,t in dl:
                x,o,u,t=[v.to(device) for v in (x,o,u,t)]
                logits=model(x,o,u);loss+=criterion(logits,t).item()*len(t);count+=len(t)
                targets.extend(t.cpu().tolist());pred.extend(logits.argmax(1).cpu().tolist())
        return loss/count,metrics(np.array(targets),np.array(pred)),targets,pred
    best=-1;stale=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();total=0;count=0
        for x,o,u,t in train:
            x,o,u,t=[v.to(device) for v in (x,o,u,t)]
            optimizer.zero_grad();loss=criterion(model(x,o,u),t)
            if not torch.isfinite(loss):raise ValueError('Loss no finita')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=loss.item()*len(t);count+=len(t)
        vl,m,_,_=evaluate(validation)
        row=dict(epoch=epoch,train_loss=total/count,val_loss=vl,val_macro_f1=m['macro_f1_34'],val_accuracy=m['accuracy'])
        history.append(row);print(row,flush=True)
        with (args.salida/'historial.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        if m['macro_f1_34']>best:
            best=m['macro_f1_34'];stale=0
            torch.save(model.state_dict(),args.salida/'mejor_modelo.pt')
        else:stale+=1
        if stale>=args.patience:break
    model.load_state_dict(torch.load(args.salida/'mejor_modelo.pt',map_location=device,weights_only=True))
    _,vm,_,_=evaluate(validation)
    report={'best_validation':vm,'test_evaluated':False}
    if args.evaluar_test:
        _,tm,targets,pred=evaluate(loader('test'));report.update(test_evaluated=True,test=tm)
        with (args.salida/'predicciones_test.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['sample_id','y_true','y_pred']);w.writerows(zip(ids[split['test']],targets,pred))
    (args.salida/'metricas.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Las etiquetas se recuperan de las muestras y de sus índices reales.
    clases = []
    for clase_id in range(34):
        nombres = {str(s).split('/')[1] for s in ids[y == clase_id]}
        if len(nombres) != 1:
            raise ValueError(f'Etiquetas ambiguas para clase {clase_id}')
        clases.append(nombres.pop())
    exportar(report, args.salida, clases)
    print('Finalizado. Test evaluado:',args.evaluar_test,flush=True)

if __name__=='__main__':main()
