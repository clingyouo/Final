"""Create immutable-index random and full-scaffold splits from frozen clean.csv."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

root=Path(__file__).resolve().parents[1]
source=root/'data/processed/clean.csv'
expected='e13ac708fd375e656cac445c289ba76af110dbbba4cba58b9905634418f648b4'
digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest==expected, 'Frozen clean.csv hash changed'
d=pd.read_csv(source)
assert len(d)==6889 and d.canonical_smiles.is_unique and d.PAMPA_clipped.notna().all()
assert d.PAMPA_clipped.between(-8,-4).all()
scaff=[]
for s in d.canonical_smiles:
    mol=Chem.MolFromSmiles(s)
    assert mol is not None
    scaff.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False))
assert np.array_equal(np.asarray(scaff),d.Murcko_Scaffold.to_numpy())
n=len(d); desired=np.array([int(.8*n),int(.1*n),n-int(.8*n)-int(.1*n)])
names=['train','val','test']
root_splits=root/'splits'; out=root/'results/split_analysis'
destinations=[root_splits/k/f'seed_{seed}' for k in ('random','scaffold') for seed in (0,1,2)]
if any(x.exists() for x in destinations) or out.exists():
    raise FileExistsError('Existing split or split-analysis output; refusing overwrite')
out.mkdir(parents=True)
all_stats=[]
group_map={s:np.flatnonzero(np.asarray(scaff)==s) for s in set(scaff)}
largest_scaffold=max(group_map,key=lambda s:len(group_map[s]))
assert len(group_map[largest_scaffold])==1101
for kind in ('random','scaffold'):
    for seed in (0,1,2):
        rng=np.random.default_rng(seed)
        if kind=='random':
            perm=rng.permutation(n)
            parts=[perm[:desired[0]],perm[desired[0]:desired[0]+desired[1]],perm[desired[0]+desired[1]:]]
        else:
            groups=list(group_map.items()); rng.shuffle(groups)
            groups.sort(key=lambda x:len(x[1]),reverse=True)
            buckets=[[],[],[]]; sizes=np.zeros(3,dtype=int)
            for scaffold,indices in groups:
                m=len(indices)
                # Minimize aggregate distance to target capacities, keeping each group whole.
                scores=[sum(abs((sizes+np.eye(3,dtype=int)[j]*m)-desired)) for j in range(3)]
                choice=min(range(3),key=lambda j:(scores[j],sizes[j]/desired[j],j))
                buckets[choice].append(indices); sizes[choice]+=m
            parts=[np.concatenate(b) if b else np.array([],dtype=int) for b in buckets]
        sets=[set(map(int,x)) for x in parts]
        assert len(set.union(*sets))==n and sum(map(len,sets))==n
        assert all(not (sets[i]&sets[j]) for i in range(3) for j in range(i+1,3))
        sc_sets=[{scaff[i] for i in indices} for indices in parts]
        if kind=='scaffold':
            assert all(not (sc_sets[i]&sc_sets[j]) for i in range(3) for j in range(i+1,3))
        folder=root_splits/kind/f'seed_{seed}'; folder.mkdir(parents=True)
        largest_location=names[next(i for i,indices in enumerate(parts) if set(group_map[largest_scaffold])<=set(indices))] if kind=='scaffold' else 'not_grouped'
        metadata={'dataset_version':'clean-v1-2026-11-04','dataset_path':'data/processed/clean.csv','dataset_sha256':digest,
                  'target_field':'PAMPA_clipped','raw_target_field':'PAMPA_raw','split_kind':kind,'seed':seed,
                  'algorithm':'numpy.default_rng.permutation' if kind=='random' else 'seeded-shuffle, descending scaffold size, minimum total absolute target-count error',
                  'requested_ratio':[.8,.1,.1],'actual_counts':{},'actual_ratio':{},
                  'largest_scaffold_count':len(group_map[largest_scaffold]),'largest_scaffold_set':largest_location,
                  'index_definition':'zero-based row index in clean.csv after header; no augmentation before split',
                  'leakage_checks':{'sample_disjoint':True,'complete_coverage':True,'scaffold_disjoint':kind=='scaffold'}}
        for name,indices,sset in zip(names,parts,sc_sets):
            indices=np.sort(indices)
            pd.DataFrame({'row_index':indices}).to_csv(folder/f'{name}.csv',index=False)
            v=d.iloc[indices].PAMPA_clipped
            metadata['actual_counts'][name]=len(indices);metadata['actual_ratio'][name]=len(indices)/n
            all_stats.append({'split_type':kind,'seed':seed,'set':name,'n':len(indices),'fraction':len(indices)/n,
                              'pampa_mean':v.mean(),'pampa_std':v.std(ddof=1),'pampa_min':v.min(),'pampa_max':v.max(),
                              'scaffold_count':len(sset),'largest_scaffold_set':largest_location,'dataset_sha256':digest})
        (folder/'split_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
pd.DataFrame(all_stats).to_csv(out/'split_statistics.csv',index=False)
# Pillow is already present in the existing environment; render a compact distribution figure without matplotlib.
from PIL import Image, ImageDraw, ImageFont
image=Image.new('RGB',(1500,900),'white'); draw=ImageDraw.Draw(image)
colors={'train':'#277DA1','val':'#F9A03F','test':'#9A6DD7'}
bins=np.linspace(-8,-4,21)
for k,(kind,seed) in enumerate([('random',0),('scaffold',0),('scaffold',1)]):
    x0=85+490*k; y0=115; h=570; w=390
    draw.text((x0,y0-45),f'{kind} seed {seed}',fill='black')
    draw.line((x0,y0+h,x0+w,y0+h),fill='black',width=2)
    draw.line((x0,y0,x0,y0+h),fill='black',width=2)
    folder=root_splits/kind/f'seed_{seed}'
    hist=[]
    for name in names:
        ix=pd.read_csv(folder/f'{name}.csv').row_index.to_numpy()
        counts,_=np.histogram(d.iloc[ix].PAMPA_clipped,bins=bins)
        hist.append(counts/len(ix))
    ymax=max(max(v) for v in hist)*1.1
    for j,name in enumerate(names):
        for b,value in enumerate(hist[j]):
            bx=x0+b*19.5+j*5; top=y0+h-int(value/ymax*(h-25))
            draw.rectangle((bx,top,bx+5,y0+h),fill=colors[name])
    draw.text((x0,y0+h+12),'PAMPA clipped: -8 to -4',fill='black')
draw.text((85,45),'Target distribution by split (normalized within each set)',fill='black')
for j,name in enumerate(names):
    draw.rectangle((105+j*160,810,125+j*160,830),fill=colors[name]);draw.text((132+j*160,810),name,fill='black')
image.save(out/'target_distribution.png')
print(pd.DataFrame(all_stats).to_string(index=False))
