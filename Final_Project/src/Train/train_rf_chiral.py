"""Chirality-aware Morgan fingerprint RF ablation; validation selects settings."""
from pathlib import Path
import json, logging
import numpy as np
from train_common import arguments,load,metrics,output_dir,save_evaluation,TARGET

def main():
    args=arguments('rf_chiral.json'); cfg,df,ix,digest=load(args)
    if args.smoke_test:
        print(f'RF smoke PASS: {len(df)} rows; {[len(ix[k]) for k in ix]}'); return
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    folder=output_dir('rf_chiral',args)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=cfg['radius'],fpSize=cfg['fingerprint_bits'],includeChirality=True)
    features=[]
    for smi in df.canonical_smiles:
        arr=np.zeros(cfg['fingerprint_bits'],dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(smi)),arr)
        features.append(arr)
    x=np.asarray(features); y=df[TARGET].to_numpy()
    best=None
    for trees in cfg['n_estimators']:
        for leaf in cfg['min_samples_leaf']:
            model=RandomForestRegressor(n_estimators=trees,min_samples_leaf=leaf,n_jobs=cfg['n_jobs'],random_state=args.seed)
            model.fit(x[ix['train']],y[ix['train']])
            score=metrics(y[ix['val']],model.predict(x[ix['val']]))['MAE']
            logging.info('n_estimators=%s min_samples_leaf=%s val_MAE=%s',trees,leaf,score)
            if best is None or score<best[0]: best=(score,trees,leaf,model)
    _,trees,leaf,model=best
    joblib.dump(model,folder/'model.joblib')
    save_evaluation(folder,df,ix,{k:model.predict(x[v]) for k,v in ix.items()},cfg,digest,
                    {'n_estimators':trees,'min_samples_leaf':leaf,'validation_MAE':best[0],
                     'note':'RF has no native early stopping; validation selects an independently trained model'})

if __name__=='__main__': main()
