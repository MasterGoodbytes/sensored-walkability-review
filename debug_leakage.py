import os, re, json, unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
np.random.seed(SEED)

DATA_DIR = 'model/dataset'
LABEL_MAP = {'Not related': 0, 'Sensing of walkability dimension': 1}
DUP_THRESHOLD = 0.80
SPLIT_SIZES = (0.70, 0.15, 0.15)

SOURCES = {
    'dataset': f'{DATA_DIR}/dataset.json',
    'contrastive': f'{DATA_DIR}/contrastive.json',
    'adaptive': f'{DATA_DIR}/adaptive.json',
    'expansion': f'{DATA_DIR}/expansion.json',
    'real': f'{DATA_DIR}/real.json',
}

def load_json_file(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return json.load(f)

def normalize_text(t):
    t = unicodedata.normalize('NFKD', str(t)).lower()
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

frames=[]
for src,path in SOURCES.items():
    if os.path.exists(path):
        d=pd.DataFrame(load_json_file(path))
        d['source']=src
        frames.append(d)
        print('loaded', src, len(d))

if not frames:
    raise SystemExit('No files')


df=pd.concat(frames, ignore_index=True)
df=df[['text','label','source']].dropna(subset=['text','label']).reset_index(drop=True)
df['norm_text']=df['text'].map(normalize_text)
df=df[df['norm_text'].str.split().str.len()>=10].copy()
conflicts=df.groupby('norm_text')['label'].nunique().loc[lambda s:s>1].index.tolist()
print('conflicts', len(conflicts))
if conflicts:
    for t in conflicts[:3]:
        sub=df[df['norm_text']==t]
        print('conflict', sub[['label','source']].drop_duplicates().to_dict('records'))
    df=df[~df['norm_text'].isin(conflicts)].copy()

n_before=len(df)
df['_prio']=(df['source']=='real').astype(int)
df=df.sort_values('_prio', ascending=False).drop_duplicates('norm_text', keep='first').drop(columns=['_prio']).reset_index(drop=True)
print('dedup exact removed', n_before-len(df))
df['label_id']=df['label'].map(LABEL_MAP)
print('unmapped', df['label_id'].isna().sum())
df=df[df['label_id'].notna()].copy(); df['label_id']=df['label_id'].astype(int); df['is_real']=df['source']=='real'

class UnionFind:
    def __init__(self,n): self.p=list(range(n))
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra!=rb: self.p[rb]=ra

texts=df['norm_text'].tolist()
vec=TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
X=vec.fit_transform(texts)
S=cosine_similarity(X)
np.fill_diagonal(S,0.0)
uf=UnionFind(len(texts))
ii,jj=np.where(np.triu(S,1)>=DUP_THRESHOLD)
for a,b in zip(ii,jj): uf.union(int(a),int(b))
_,groups=np.unique([uf.find(i) for i in range(len(texts))], return_inverse=True)
df['group_id']=groups

g=df.groupby('group_id').agg(n=('label_id','size'), lab=('label_id', lambda s: int(s.mode().iloc[0])), prio=('is_real','max')).reset_index()

rng=np.random.default_rng(SEED)
train_g=[]; val_g=[]; test_g=[]
for lab,sub in g.groupby('lab'):
    sub=sub.sample(frac=1.0, random_state=int(rng.integers(1e6)))
    sub=sub.sort_values('prio', ascending=False, kind='stable')
    tot=sub['n'].sum(); q_val=SPLIT_SIZES[1]*tot; q_test=SPLIT_SIZES[2]*tot
    acc_v=0; acc_t=0
    for _,r in sub.iterrows():
        if acc_v < q_val:
            val_g.append(r['group_id']); acc_v += r['n']
        elif acc_t < q_test:
            test_g.append(r['group_id']); acc_t += r['n']
        else:
            train_g.append(r['group_id'])

train_df=df[df['group_id'].isin(train_g)].copy().reset_index(drop=True)
val_df=df[df['group_id'].isin(val_g)].copy().reset_index(drop=True)
test_df=df[df['group_id'].isin(test_g)].copy().reset_index(drop=True)
print('train', len(train_df), 'val', len(val_df), 'test', len(test_df))

v=TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
Z=v.fit_transform(train_df['norm_text'].tolist()+val_df['norm_text'].tolist())
Sx=cosine_similarity(Z[len(train_df):], Z[:len(train_df)])
print('audit max', Sx.max())
rows,cols=np.where(Sx>=DUP_THRESHOLD)
print('audit pairs', len(rows))
for i,j in zip(rows,cols):
    print('pair', i, j, 'sim', Sx[i,j])
    print('val text', val_df.iloc[i]['norm_text'])
    print('train text', train_df.iloc[j]['norm_text'])
    print('val grp', val_df.iloc[i]['group_id'], 'train grp', train_df.iloc[j]['group_id'])
    print('same group', val_df.iloc[i]['group_id']==train_df.iloc[j]['group_id'])
    break

# compare with original group-based similarity from full corpus
print('full corpus max sim', np.max(np.where(np.triu(S,1)>=DUP_THRESHOLD, S, 0)))
