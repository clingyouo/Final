"""Independently read persisted split files and verify leakage constraints."""
from pathlib import Path
import hashlib, json
import pandas as pd

root=Path(__file__).resolve().parents[1]
clean=root/'data/processed/clean.csv'
d=pd.read_csv(clean)
digest=hashlib.sha256(clean.read_bytes()).hexdigest()
for kind in ('random','scaffold'):
    for seed in (0,1,2):
        folder=root/'splits'/kind/f'seed_{seed}'
        meta=json.loads((folder/'split_metadata.json').read_text())
        assert meta['dataset_sha256']==digest
        sets=[set(pd.read_csv(folder/f'{name}.csv').row_index.astype(int)) for name in ('train','val','test')]
        assert all(min(s)>=0 and max(s)<len(d) for s in sets)
        assert len(set.union(*sets))==len(d)
        assert sum(map(len,sets))==len(d)
        assert all(not sets[i].intersection(sets[j]) for i in range(3) for j in range(i+1,3))
        scaffold_sets=[set(d.iloc[list(s)].Murcko_Scaffold) for s in sets]
        if kind=='scaffold':
            assert all(not scaffold_sets[i].intersection(scaffold_sets[j]) for i in range(3) for j in range(i+1,3))
        print(kind,seed,[len(s) for s in sets], 'coverage=PASS overlap=PASS scaffold=' + ('PASS' if kind=='scaffold' else 'not required'))
print('clean_sha256',digest)
