"""Chirality-aware Morgan fingerprint XGBoost ablation with validation-only early stopping."""
import logging
import numpy as np
from train_common import arguments,load,output_dir,save_evaluation,TARGET

def main():
    args=arguments('xgboost_chiral.json'); cfg,df,ix,digest=load(args)
    if args.smoke_test:
        print(f'XGBoost smoke PASS: {len(df)} rows; {[len(ix[k]) for k in ix]}'); return
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from xgboost import XGBRegressor
    folder=output_dir('xgboost_chiral',args)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=cfg['radius'],fpSize=cfg['fingerprint_bits'],includeChirality=True)
    features=[]
    for smi in df.canonical_smiles:
        arr=np.zeros(cfg['fingerprint_bits'],dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(smi)),arr)
        features.append(arr)
    x=np.asarray(features); y=df[TARGET].to_numpy()
    model=XGBRegressor(n_estimators=cfg['n_estimators'],max_depth=cfg['max_depth'],learning_rate=cfg['learning_rate'],
                       subsample=cfg['subsample'],colsample_bytree=cfg['colsample_bytree'],
                       objective='reg:squarederror',eval_metric='mae',early_stopping_rounds=cfg['early_stopping_rounds'],
                       random_state=args.seed,n_jobs=cfg['n_jobs'],tree_method='hist')
    model.fit(x[ix['train']],y[ix['train']],eval_set=[(x[ix['val']],y[ix['val']])],verbose=False)
    logging.info('best_iteration=%s',model.best_iteration)
    model.save_model(str(folder/'model.json'))
    save_evaluation(folder,df,ix,{k:model.predict(x[v]) for k,v in ix.items()},cfg,digest,
                    {'best_iteration':int(model.best_iteration),'early_stopping_metric':'validation MAE'})

if __name__=='__main__': main()
