import dgl
import collections
import numpy as np
import pandas as pd
from config import *
import networkx as nx
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def genus_pre_process(genus_data, scaler=None, embed=0, method='umap'):
    data = genus_data.drop(columns='Treat_time_M').rename(columns={'Treat_time_D':'time'})
    label = data.iloc[:,:2]
    feat = data.iloc[:,2:]
    if scaler: feat[:] = scaler.fit_transform(feat.values)
    
    if embed:
        if method == 'umap':
            import umap
            reducer = umap.UMAP(random_state=GLOBALSEED, n_components=embed)
            embedding = reducer.fit_transform(feat.values)
            feat = pd.DataFrame(embedding, columns=[f'x{i+1}' for i in range(embed)], index=label.index)
        elif method == 'chi2':
            from sklearn.feature_selection import SelectKBest, chi2
            feat_new = SelectKBest(chi2, k=embed).fit_transform(feat.values, label.event)
            feat = pd.DataFrame(feat_new, columns=[f'x{i+1}' for i in range(embed)], index=label.index)

    data = pd.concat([label, feat], axis=1)
    
    print('Genus Data:')
    print(data)
    print('Scaler:', scaler)
    print(f'{method}: {embed}')

    return data

def kegg_pre_process(data, scaler=None):
    data = data.transpose()
    if scaler: data[:] = scaler.fit_transform(data.values)

    print('KEGG Data:')
    print(data)
    print('Scaler:', scaler)

    return data

def clinical_pre_process(clinical, scaler=None):
    clinical.dropna(inplace=True)
    clinical.set_index('PRE', inplace=True)

    clinical['preantibio'] = (clinical['preantibio_yes,1;no,2'] == 1).astype(int)
    clinical['antibio'] = (clinical['antibio'] == 'はい').astype(int)
    clinical['postantibio'] = (clinical['postantibio_yes,1;no,2'] == 1).astype(int)
    clinical['mono'] = (clinical['mono,1;comb,2'] == 1).astype(int)
    clinical['effect_PD'] = (clinical['effect_o'] == 'PD').astype(int)
    clinical['effect_PR'] = (clinical['effect_o'] == 'PR').astype(int)
    clinical['effect_SD'] = (clinical['effect_o'] == 'SD').astype(int)
    clinical['effect_CR'] = (clinical['effect_o'] == 'CR').astype(int)
    clinical['time'] = clinical['Treat_time_D']

    clinical.drop(columns=['Treat_time_M', 'effect_o', 'preantibio_yes,1;no,2', 'postantibio_yes,1;no,2', 'mono,1;comb,2'], inplace=True)
    clinical = clinical[[
        'event', 'time', # label
        'preantibio', 'antibio', 'postantibio', 'mono', 'Responder', # binary
        'effect_PD', 'effect_PR', 'effect_SD', 'effect_CR',
        'ASV_Faith\'s_PD', 'ASV_observed', 'ASV_Shannon', 'ASV_Pielou', # values
        'KO_observed', 'KO_Shannon', 'KO_Pielou', 'Pathway_observed', 'Pathway_Shannon', 'Pathway_Pielou', 'adjusted_Pathway_Shannon', 'adjusted_Pathway_Pielou', 'Potential_compounds', 'unweighted_PCI', 'weighted_PCI'
        ]]
    
    feat = clinical.iloc[:, 11:]
    if scaler: feat[:] = scaler.fit_transform(feat.values)
    pd.options.mode.chained_assignment = None
    clinical[feat.columns] = feat

    print('Clinical Data:')
    print(clinical)
    print('Scaler:', scaler)
    return clinical

def del_topk(data, topk_list, k='all', is_rand=False):
    if is_rand:
        topk = topk_list[np.random.choice(len(topk_list), k, False)]
    else:
        topk = topk_list if k == 'all' else topk_list[:k]
    data = data.drop(topk, axis=1).iloc[:,2:]
    print('Random Delete Data:' if is_rand else 'Delete top-k Data:')
    print(data)
    print(f'{"Random" if is_rand else "Top"} {len(topk)} microbes have been removed:\n{[TAXID_NAME[i] for i in topk]}')
    return data

class Dataset:
    def __init__(self, df, df2=None) -> None:
        self.data = df
        self.interval = df2
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data.iloc[idx, 2:].to_numpy())
        ye = self.data.event[idx]

        if self.interval is not None:
            i = self.interval[idx]
            yt = np.zeros(INTERVALS)
            yt[:i-ye+1] = 1
        else:
            yt = self.data.time[idx]
        return x, yt, ye

def loader(data, yt=None, shuffle=True):
    return DataLoader(Dataset(data, yt), batch_size=BATCH, shuffle=shuffle)

genus_data = pd.read_pickle('../data/processed_data/df_inp_label_3_genus_85.pkl')
genus_human_data = pd.read_pickle('../data/processed_data/df_inp_label_3_genus_112.pkl')
genus_all_data = pd.read_pickle('../data/processed_data/df_inp_label_3_genus_254.pkl')
kegg_data = pd.read_pickle('../data/processed_data/pred_path_abun.pkl')
clinical_data = pd.read_csv('../data/raw_data/meta_data.txt')
adj_g85_data = pd.read_pickle('../data/processed_data/df_inp_graph_85.pkl')
adj_g254_data = pd.read_pickle('../data/processed_data/df_inp_graph_254.pkl')
adj_g112_data = pd.read_pickle('../data/processed_data/df_inp_graph_112.pkl')
adj_g202_data = pd.read_pickle('../data/processed_data/df_inp_graph_202.pkl')

genus_data = genus_pre_process(genus_data, MinMaxScaler(), embed=0, method='chi2')
genus_human_data = genus_pre_process(genus_human_data, MinMaxScaler(), embed=0, method='chi2')
genus_all_data = genus_pre_process(genus_all_data, MinMaxScaler(), embed=0, method='chi2')
kegg_data = kegg_pre_process(kegg_data, MinMaxScaler())
clinical_data = clinical_pre_process(clinical_data, MinMaxScaler())
# genus_topk = del_topk(genus_data, TOP30)
# genus_rand = del_topk(genus_data, adj_g85_data.columns, k=30, is_rand=True)

df_et_cli = clinical_data.copy()
df_et_g85 = genus_data.copy()
df_et_g112 = genus_human_data.copy()
df_et_cli_g85 = clinical_data.merge(genus_data.iloc[:,2:], left_index=True, right_index=True)
df_et_cli_g112 = clinical_data.merge(genus_human_data.iloc[:,2:], left_index=True, right_index=True)
df_et_cli_kegg_g85 = clinical_data.merge(kegg_data, left_index=True, right_index=True).merge(genus_data.iloc[:,2:], left_index=True, right_index=True)
df_et_cli_g254 = clinical_data.merge(genus_all_data.iloc[:,2:], left_index=True, right_index=True)
# df_et_cli_g85_del_topk = clinical_data.merge(genus_topk, left_index=True, right_index=True)
# df_et_cli_g85_del_rand = clinical_data.merge(genus_rand, left_index=True, right_index=True)

g85_und = dgl.from_networkx(nx.from_numpy_array(adj_g85_data.to_numpy())) # Undirected graph
g85_di = dgl.add_self_loop(dgl.from_networkx(nx.from_numpy_array(adj_g85_data.to_numpy(), create_using=nx.DiGraph))) # There are 0-in-degree nodes in the graph
# g85_del_topk = dgl.from_networkx(nx.from_numpy_array(adj_g85_data.loc[genus_topk.columns, genus_topk.columns].to_numpy()))
# g85_del_rand = dgl.from_networkx(nx.from_numpy_array(adj_g85_data.loc[genus_rand.columns, genus_rand.columns].to_numpy()))
g254_und = dgl.add_self_loop(dgl.from_networkx(nx.from_numpy_array(adj_g254_data.to_numpy())))
g112_und = dgl.from_networkx(nx.from_numpy_array(adj_g112_data.to_numpy()))
g202_und = dgl.add_self_loop(dgl.from_networkx(nx.from_numpy_array(adj_g202_data.to_numpy())))
g202_idn = dgl.from_networkx(nx.from_numpy_array(np.identity(202)))


def batch2_pre_process(data, scaler=None):
    if scaler: data.iloc[:,-5:] = scaler.fit_transform(data.iloc[:,-5:].values)

    print('Batch2 Data:')
    print(data)
    print('Scaler:', scaler)

    return data

batch2_data = pd.read_pickle('../data/processed_data/df_inp_batch2_321.pkl')
df_et_cli_b2 = batch2_data.rename(columns={'OS_E':'event', 'OS_D':'time'})[['event', 'time']+batch2_data.columns[7:].to_list()]
df_et_cli_b2 = batch2_pre_process(df_et_cli_b2, MinMaxScaler())

cancer = df_et_cli_b2.iloc[:,2:18]
clinical_data[cancer.columns] = 0
ints = clinical_data.index.intersection(cancer.index)
clinical_data.loc[ints, cancer.columns] = cancer.loc[ints]

df_et_can_cli = clinical_data.copy()
df_et_can_cli_g112 = clinical_data.merge(genus_human_data.iloc[:,2:], left_index=True, right_index=True)

def batch3_pre_process(data, scaler=None):
    if scaler: data.iloc[:,29:] = scaler.fit_transform(data.iloc[:,29:].values)

    data.age /= 100

    print('Batch3 Data:')
    print(data)
    print('Scaler:', scaler)

    return data

b3_cli_data = pd.read_pickle('../data/processed_data/df_inp_b3_cli_502_ori.pkl')
b3_microbes_data = pd.read_pickle('../data/processed_data/df_inp_microbes_202.pkl')
b3_OS_cli_data = b3_cli_data.copy().rename(columns={'OS_E':'event', 'OS_D':'time'})[['event', 'time']+b3_cli_data.columns[6:].to_list()]
b3_PFS_cli_data = b3_cli_data.copy().rename(columns={'PFS_E':'event', 'PFS_D':'time'})[['event', 'time']+b3_cli_data.columns[6:].to_list()]
b3_TTF_cli_data = b3_cli_data.copy().rename(columns={'TTF_E':'event', 'TTF_D':'time'})[['event', 'time']+b3_cli_data.columns[6:].to_list()]

df_et_OS_cli_b3 = batch3_pre_process(b3_OS_cli_data.copy(), MinMaxScaler())
df_et_OS_cli_g202_b3 = batch3_pre_process(b3_OS_cli_data.copy().merge(b3_microbes_data, left_index=True, right_index=True), MinMaxScaler())
df_et_PFS_cli_b3 = batch3_pre_process(b3_PFS_cli_data.copy(), MinMaxScaler())
df_et_PFS_cli_g202_b3 = batch3_pre_process(b3_PFS_cli_data.copy().merge(b3_microbes_data, left_index=True, right_index=True), MinMaxScaler())
df_et_TTF_cli_b3 = batch3_pre_process(b3_TTF_cli_data.copy(), MinMaxScaler())
df_et_TTF_cli_g202_b3 = batch3_pre_process(b3_TTF_cli_data.copy().merge(b3_microbes_data, left_index=True, right_index=True), MinMaxScaler())

df_et_OS_cli_g202_b3_nocancer = df_et_OS_cli_g202_b3.drop(labels=df_et_OS_cli_g202_b3.columns[11:29], axis=1)
df_et_PFS_cli_g202_b3_nocancer = df_et_PFS_cli_g202_b3.drop(labels=df_et_PFS_cli_g202_b3.columns[11:29], axis=1)

df_et_OS_g202_b3 = df_et_OS_cli_g202_b3[['event', 'time']+b3_microbes_data.columns.to_list()]
df_et_PFS_g202_b3 = df_et_PFS_cli_g202_b3[['event', 'time']+b3_microbes_data.columns.to_list()]
df_et_TTF_g202_b3 = df_et_TTF_cli_g202_b3[['event', 'time']+b3_microbes_data.columns.to_list()]

_cols = df_et_OS_cli_g202_b3.columns[:11].to_list() + df_et_OS_cli_g202_b3.columns[28:].to_list()
df_OS_malignant_melanoma = df_et_OS_cli_g202_b3[df_et_OS_cli_g202_b3['悪性黒色腫']==1][_cols].copy() # 55
df_OS_head_neck = df_et_OS_cli_g202_b3[df_et_OS_cli_g202_b3['頭頸部癌']==1][_cols].copy() # 106
df_OS_renal_cell = df_et_OS_cli_g202_b3[df_et_OS_cli_g202_b3['腎細胞癌']==1][_cols].copy() # 75
df_PFS_malignant_melanoma = df_et_PFS_cli_g202_b3[df_et_PFS_cli_g202_b3['悪性黒色腫']==1][_cols].copy() # 55
df_PFS_head_neck = df_et_PFS_cli_g202_b3[df_et_PFS_cli_g202_b3['頭頸部癌']==1][_cols].copy() # 106
df_PFS_renal_cell = df_et_PFS_cli_g202_b3[df_et_PFS_cli_g202_b3['腎細胞癌']==1][_cols].copy() # 75
df_PFS_bladder = df_et_PFS_cli_g202_b3[df_et_PFS_cli_g202_b3['腎盂・尿管・膀胱癌']==1][_cols].copy() # 74

with open('../data/intermediate_data/b3_taxid_lineage.txt', 'r') as f:
    taxid2phylum = {}
    taxid2lineage = {}
    for line in f:
        if (arr := line.strip().split()):
            taxid2phylum[arr[0]] = arr[1].split('p__')[1].split(';')[0]
            taxid2lineage[arr[0]] = arr[1]
taxoNN_cluster_num = 4
taxoNN_microbes_p = [taxid2phylum[str(taxid)] for taxid in b3_microbes_data.columns]
taxoNN_p2i = {item[0]:i for i, item in enumerate(collections.Counter(taxoNN_microbes_p).most_common(taxoNN_cluster_num-1))}
taxoNN_microbes_cluster = [taxoNN_p2i.get(p, taxoNN_cluster_num-1) for p in taxoNN_microbes_p]

metaDR_microbes_tree = NCBITaxa().get_topology(b3_microbes_data.columns)
order = []
for node in metaDR_microbes_tree.traverse(strategy='levelorder'):
    if node.is_leaf(): order.append(int(node.name))
postorder = []
for node in metaDR_microbes_tree.traverse(strategy='postorder'):
    if node.is_leaf(): postorder.append(int(node.name))
print(f'Unused microbes (MetaDR): {set(b3_microbes_data.columns) - set(order)}')
metaDR_kl_idx = [b3_microbes_data.columns.to_list().index(taxid) for taxid in order]
metaDR_kp_idx = [b3_microbes_data.columns.to_list().index(taxid) for taxid in postorder]

# breakpoint()

# dt = (data.time.max()+-data.time.max()%5) / INTERVALS
# time_range = np.linspace(0, data.time.max()+-data.time.max()%5, INTERVALS+1)
# time_i = [i if time_range[i] < d <= time_range[i+1] else max(i-1,0) for i, d in zip((data.time // dt).astype(int), data.time)]
# onehot = np.zeros((len(data), INTERVALS), dtype=int)
# for i, j in enumerate(time_i): onehot[i, :j] = 1
# data_interval = pd.DataFrame(onehot, columns=time_range[1:], index=data.index)
