"""Shared frozen-data loading, split checks, metrics, and result writing."""
from __future__ import annotations
import argparse, hashlib, json, logging
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_SHA='e13ac708fd375e656cac445c289ba76af110dbbba4cba58b9905634418f648b4'
TARGET='PAMPA_clipped'

def arguments(default_config):
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=str(ROOT/'configs'/default_config))
    ap.add_argument('--split-type',choices=['random','scaffold'],required=True)
    ap.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    ap.add_argument('--smoke-test',action='store_true',help='Read and validate inputs only; no model fitting or outputs')
    return ap.parse_args()

def load(args):
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if cfg['dataset']!='data/processed/clean.csv' or cfg['target']!=TARGET:
        raise ValueError('Configuration must use frozen clean.csv and PAMPA_clipped')
    path=ROOT/cfg['dataset']; digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA: raise ValueError('Frozen dataset SHA256 mismatch')
    df=pd.read_csv(path)
    if len(df)!=6889 or df.canonical_smiles.duplicated().any() or df[TARGET].isna().any():
        raise ValueError('Frozen dataset validation failed')
    folder=ROOT/'splits'/args.split_type/f'seed_{args.seed}'
    meta=json.loads((folder/'split_metadata.json').read_text())
    if meta['dataset_sha256']!=digest or meta['target_field']!=TARGET or meta['seed']!=args.seed or meta['split_kind']!=args.split_type:
        raise ValueError('Split metadata mismatch')
    indices={name:pd.read_csv(folder/f'{name}.csv').row_index.to_numpy(dtype=int) for name in ('train','val','test')}
    flat=np.concatenate(list(indices.values()))
    if len(flat)!=len(df) or not np.array_equal(np.sort(flat),np.arange(len(df))):
        raise ValueError('Split indices do not form a disjoint full partition')
    if args.split_type=='scaffold':
        groups=[set(df.iloc[ix].Murcko_Scaffold) for ix in indices.values()]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
            raise ValueError('Scaffold leakage')
    return cfg,df,indices,digest

def metrics(y,pred):
    y=np.asarray(y,dtype=float).reshape(-1); pred=np.asarray(pred,dtype=float).reshape(-1)
    if len(y)!=len(pred) or not np.isfinite(pred).all(): raise ValueError('Invalid predictions')
    residual=y-pred
    r2=1-float(np.dot(residual,residual)/np.sum((y-y.mean())**2)) if np.var(y)>0 else None
    pearson=float(np.corrcoef(y,pred)[0,1]) if np.std(y)>0 and np.std(pred)>0 else None
    return {'MAE':float(np.mean(np.abs(residual))),'RMSE':float(np.sqrt(np.mean(residual**2))),'R2':r2,'Pearson_R':pearson}

def output_dir(model,args):
    folder=ROOT/'results'/'models'/model/args.split_type/f'seed_{args.seed}'
    if folder.exists(): raise FileExistsError(f'Run output already exists: {folder}')
    folder.mkdir(parents=True)
    logging.basicConfig(filename=folder/'train.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    return folder

def save_evaluation(folder,df,indices,predictions,config,digest,selected):
    report={'dataset_sha256':digest,'target':TARGET,'selected_by_validation':selected,'config':config,'metrics':{}}
    for name in ('train','val','test'):
        ix=indices[name]; pred=np.asarray(predictions[name]).reshape(-1)
        report['metrics'][name]=metrics(df.iloc[ix][TARGET],pred)
        pd.DataFrame({'row_index':ix,'canonical_smiles':df.iloc[ix].canonical_smiles.to_numpy(),
                      'y_true':df.iloc[ix][TARGET].to_numpy(),'y_pred':pred}).to_csv(folder/f'predictions_{name}.csv',index=False)
    (folder/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    logging.info('completed metrics=%s',report['metrics'])
