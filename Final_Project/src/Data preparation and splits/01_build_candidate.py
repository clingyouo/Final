"""Build a traceable PAMPA candidate from immutable raw and author reference files."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

root=Path(__file__).resolve().parents[1]
rawpath=root/'data/raw/CycPeptMPDB_Peptide_All.csv'
authpath=root/'references/peptide_used_author.csv'
out=root/'data/processed'
reportpath=root/'results/final_dataset_check.md'
names=['clean_candidate.csv','duplicate_resolution.csv','clipping_log.csv','dataset_metadata.json']
for path in [*(out/n for n in names),reportpath]:
    if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
raw=pd.read_csv(rawpath,low_memory=False)
author=pd.read_csv(authpath,low_memory=False)
assert raw.CycPeptMPDB_ID.is_unique
p=raw[raw.PAMPA.notna()].copy()
def canonical(s):
    m=Chem.MolFromSmiles(s) if pd.notna(s) else None
    return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) if m else None
p['canonical_smiles']=[canonical(x) for x in p.SMILES]
author['canonical_smiles']=[canonical(x) for x in author.SMILES]
assert p.canonical_smiles.notna().all() and author.canonical_smiles.notna().all()
assert len(p)==6941 and p.canonical_smiles.nunique()==6889
raw_by_id=raw.set_index('CycPeptMPDB_ID')
author['raw_id_present']=author.ID.isin(raw_by_id.index)
author['id_smiles_match']=[bool(i in raw_by_id.index and canonical(raw_by_id.at[i,'SMILES'])==s) for i,s in zip(author.ID,author.canonical_smiles)]
author['id_pampa_match']=[bool(i in raw_by_id.index and pd.notna(v) and pd.notna(raw_by_id.at[i,'PAMPA']) and np.isclose(v,raw_by_id.at[i,'PAMPA'],atol=1e-9)) for i,v in zip(author.ID,author.PAMPA)]
author['id_permeability_match']=[bool(i in raw_by_id.index and np.isclose(v,raw_by_id.at[i,'Permeability'],atol=1e-9)) for i,v in zip(author.ID,author.Objective_variable)]
author_p=author[author.PAMPA.notna()].copy()
assert len(author_p)==6889

rows=[]; candidates=[]
author_by_structure={s:g for s,g in author_p.groupby('canonical_smiles')}
for structure,g in p.groupby('canonical_smiles',sort=False):
    g=g.sort_values(['Year','CycPeptMPDB_ID'],ascending=[False,True])
    ag=author_by_structure.get(structure)
    matched=[]
    if ag is not None:
        matched=[int(i) for i in ag.ID if int(i) in set(g.CycPeptMPDB_ID)]
    if len(matched)==1:
        chosen=g[g.CycPeptMPDB_ID.eq(matched[0])].iloc[0]
        value=float(chosen.PAMPA); method='author_record'
        contributing=[int(chosen.CycPeptMPDB_ID)]
    else:
        maxyear=int(g.Year.max()); newest=g[g.Year.eq(maxyear)]
        value=float(newest.PAMPA.median())
        chosen=newest.iloc[0]
        method='latest_year_median' if newest.PAMPA.nunique()>1 else 'latest_year_same_value'
        contributing=[int(x) for x in newest.CycPeptMPDB_ID]
    candidate={'canonical_smiles':structure,'PAMPA_raw':value,'PAMPA_clipped':float(np.clip(value,-8,-4)),
               'Source':str(chosen.Source) if len(contributing)==1 else '; '.join(sorted(g[g.CycPeptMPDB_ID.isin(contributing)].Source.astype(str).unique())),
               'Year':int(chosen.Year),'CycPeptMPDB_ID':int(chosen.CycPeptMPDB_ID) if len(contributing)==1 else '; '.join(map(str,contributing)),
               'Murcko_Scaffold':MurckoScaffold.MurckoScaffoldSmiles(smiles=structure),
               'resolution_method':method,'measurement_count':len(g)}
    candidates.append(candidate)
    if len(g)>1:
        rows.append({'canonical_smiles':structure,'record_count':len(g),'method':method,
                     'all_ids':json.dumps([int(x) for x in g.CycPeptMPDB_ID]),
                     'all_sources':json.dumps([str(x) for x in g.Source]),
                     'all_years':json.dumps([int(x) for x in g.Year]),
                     'all_pampa_raw':json.dumps([float(x) for x in g.PAMPA]),
                     'author_matched_ids':json.dumps(matched),
                     'contributing_ids':json.dumps(contributing),
                     'selected_raw_value':value,'max_year_tie':int(g.Year.eq(g.Year.max()).sum())>1})
clean=pd.DataFrame(candidates).sort_values('canonical_smiles').reset_index(drop=True)
dup=pd.DataFrame(rows).sort_values('canonical_smiles')
clips=clean[clean.PAMPA_raw.ne(clean.PAMPA_clipped)][['canonical_smiles','CycPeptMPDB_ID','Source','Year','PAMPA_raw','PAMPA_clipped','resolution_method']].copy()
clips['reason']=np.where(clips.PAMPA_raw.lt(-8),'below_-8','above_-4')
disagreement=p[p.Permeability.ne(p.PAMPA)].copy()
disagreement_author=author_p[author_p.ID.isin(disagreement.CycPeptMPDB_ID)]
summary={'raw_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),
         'author_url':'https://raw.githubusercontent.com/akiyamalab/cycpeptmp/main/desc/peptide_used.csv',
         'author_sha256':hashlib.sha256(authpath.read_bytes()).hexdigest(),'author_bytes':authpath.stat().st_size,
         'author_columns':list(author.columns[:32]),'author_rows':len(author),'author_unique_id':int(author.ID.nunique()),
         'author_pampa_rows':len(author_p),'author_id_present':int(author.raw_id_present.sum()),
         'author_id_smiles_match':int(author.id_smiles_match.sum()),
         'author_id_pampa_match':int(author.id_pampa_match.sum()),
         'author_id_permeability_match':int(author.id_permeability_match.sum()),
         'author_pampa_structure_overlap':len(set(author_p.canonical_smiles)&set(p.canonical_smiles)),
         'raw_pampa_rows':len(p),'candidate_rows':len(clean),'duplicate_groups':len(dup),
         'author_selected_groups':int(clean.resolution_method.eq('author_record').sum()),
         'fallback_groups':int(clean.resolution_method.ne('author_record').sum()),
         'author_disagreement_rows':len(disagreement_author),
         'author_disagreement_pampa_match':int(disagreement_author.id_pampa_match.sum()),
         'author_disagreement_objective_equals_raw_permeability':int(disagreement_author.id_permeability_match.sum()),
         'clipped_low':int(clips.reason.eq('below_-8').sum()),'clipped_high':int(clips.reason.eq('above_-4').sum()),
         'invalid_smiles':int(clean.canonical_smiles.map(canonical).isna().sum()),
         'duplicate_canonical':int(clean.canonical_smiles.duplicated().sum()),
         'missing_target':int(clean.PAMPA_raw.isna().sum()),
         'out_of_range':int((~clean.PAMPA_clipped.between(-8,-4)).sum()),
         'chiral_marked_structures':int(clean.canonical_smiles.str.contains('@',regex=False).sum())}
out.mkdir(parents=True,exist_ok=True)
clean.to_csv(out/names[0],index=False,encoding='utf-8-sig')
dup.to_csv(out/names[1],index=False,encoding='utf-8-sig')
clips.to_csv(out/names[2],index=False,encoding='utf-8-sig')
(out/names[3]).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
