"""Chemprop 2.3.x official directed-bond MPNN regression workflow."""
import logging
import numpy as np
from train_common import arguments,load,output_dir,save_evaluation,TARGET

def main():
    args=arguments('chemprop.json'); cfg,df,ix,digest=load(args)
    if args.smoke_test:
        print(f'Chemprop smoke PASS: {len(df)} rows; {[len(ix[k]) for k in ix]}'); return
    import torch
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
    from chemprop import data,featurizers,models,nn
    pl.seed_everything(args.seed,workers=True)
    folder=output_dir('chemprop',args)
    def datapoints(indices):
        return [data.MoleculeDatapoint.from_smi(df.iloc[i].canonical_smiles,
                np.array([float(df.iloc[i][TARGET])])) for i in indices]
    featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer()
    datasets={name:data.MoleculeDataset(datapoints(indices),featurizer) for name,indices in ix.items()}
    scaler=datasets['train'].normalize_targets()
    datasets['val'].normalize_targets(scaler)
    loaders={name:data.build_dataloader(dataset,batch_size=cfg['batch_size'],num_workers=cfg['num_workers'],shuffle=name=='train')
             for name,dataset in datasets.items()}
    predictor=nn.RegressionFFN(output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    model=models.MPNN(nn.BondMessagePassing(),nn.MeanAggregation(),predictor,True,[nn.metrics.MAE()])
    checkpoint=ModelCheckpoint(dirpath=folder/'checkpoints',filename='best',monitor='val_loss',mode='min',save_top_k=1)
    stopper=EarlyStopping(monitor='val_loss',mode='min',patience=cfg['patience'])
    trainer=pl.Trainer(default_root_dir=folder,max_epochs=cfg['max_epochs'],accelerator=cfg['accelerator'],devices=1,
                       callbacks=[checkpoint,stopper],logger=False,enable_progress_bar=False,
                       log_every_n_steps=1,deterministic=cfg['deterministic'])
    logging.info('fitting official Chemprop BondMessagePassing D-MPNN; checkpoint selection by val_loss')
    trainer.fit(model,loaders['train'],loaders['val'])
    if not checkpoint.best_model_path: raise RuntimeError('No best validation checkpoint')
    best=models.MPNN.load_from_checkpoint(checkpoint.best_model_path,map_location='cpu')
    predictions={}
    for name in ('train','val','test'):
        predict_loader=data.build_dataloader(datasets[name],batch_size=cfg['batch_size'],num_workers=cfg['num_workers'],shuffle=False)
        outputs=trainer.predict(best,predict_loader)
        predictions[name]=np.concatenate([t.detach().cpu().numpy().reshape(-1) for t in outputs])
    save_evaluation(folder,df,ix,predictions,cfg,digest,
                    {'checkpoint':checkpoint.best_model_path,'best_val_loss':float(checkpoint.best_model_score),
                     'early_stopping':'validation val_loss'})

if __name__=='__main__': main()
