from config import *
set_metaverse_seed(GLOBALSEED)
import dataset
import baselines

import dgl
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

def set_mode(dataset, mode):
    if mode == 'cli_g85':
        data = dataset.df_et_cli_g85
        g = dataset.g85_und
        gn = 'UnD'
    elif mode == 'cli':
        data = dataset.df_et_cli
        g = None
        gn = None
    elif mode == 'g85':
        data = dataset.df_et_g85
        g = dataset.g85_und
        gn = 'UnD'
    elif mode == 'cli_g254':
        data = dataset.df_et_cli_g254
        g = dataset.g254_und
        gn = 'UnD'
    elif mode == 'b2_cli':
        data = dataset.df_et_cli_b2
        g = None
        gn = None
    elif mode == 'can_cli_g112':
        data = dataset.df_et_can_cli_g112
        g = dataset.g112_und
        gn = 'UnD'
    elif mode == 'can_cli':
        data = dataset.df_et_can_cli
        g = None
        gn = None
    elif mode == 'g112':
        data = dataset.df_et_g112
        g = dataset.g112_und
        gn = 'UnD'
    elif mode == 'b3_OS_cli':
        data = dataset.df_et_OS_cli_b3
        g = None
        gn = None
    elif mode == 'b3_OS_cli_g202':
        data = dataset.df_et_OS_cli_g202_b3
        g = dataset.g202_und
        gn = 'UnD'
    elif mode == 'b3_OS_g202':
        data = dataset.df_et_OS_g202_b3
        g = dataset.g202_und
        gn = 'UnD'
    elif mode == 'b3_PFS_cli':
        data = dataset.df_et_PFS_cli_b3
        g = None
        gn = None
    elif mode == 'b3_PFS_cli_g202':
        data = dataset.df_et_PFS_cli_g202_b3
        g = dataset.g202_und
        gn = 'UnD'
    elif mode == 'b3_PFS_cli_g202_idn':
        data = dataset.df_et_PFS_cli_g202_b3
        g = dataset.g202_idn
        gn = 'Idn'
    elif mode == 'b3_PFS_g202':
        data = dataset.df_et_PFS_g202_b3
        g = dataset.g202_und
        gn = 'UnD'
    elif mode == 'b3_TTF_cli':
        data = dataset.df_et_TTF_cli_b3
        g = None
        gn = None
    elif mode == 'b3_TTF_cli_g202':
        data = dataset.df_et_TTF_cli_g202_b3
        g = dataset.g202_und
        gn = 'UnD'
    elif mode == 'b3_TTF_g202':
        data = dataset.df_et_TTF_g202_b3
        g = dataset.g202_und
        gn = 'UnD'
    elif mode == 'b3_OS_cancer_mm':
        data = dataset.df_malignant_melanoma
        g = dataset.g202_und
        gn = 'UnD'
    elif mode == 'b3_OS_cancer_hn':
        data = dataset.df_head_neck
        g = dataset.g202_und
        gn = 'UnD'
    elif mode == 'b3_OS_cancer_rc':
        data = dataset.df_renal_cell
        g = dataset.g202_und
        gn = 'UnD'
    return data, g, gn

def run_benchmark(dataset, mode='cli_g85', trial=TRIAL):
    data, g, gn = set_mode(dataset, mode)
    out = f'{mode}_t{trial}_{CONFIG}.csv'

    print(f'params:')
    params = [('max epochs', EPOCHS), ('batch', BATCH), ('opt', 'Adam'), ('lr', LR), ('scheduler', 'EarlyStop'), ('dropout', DROPOUT),
    ('SAG pool_ratio', SAGPOOL_RATIO), ('MVP pool_ratio', MVPOOL_RATIO), ('CPH penalty', CPH_PENALTY), ('RSF n_estimator', RSF_N_ESTIMATORS),
    ('mode', mode), ('graph', str(g).replace('\n','').replace(' '*5,'')), ('graph name', gn), ('out', out)]
    for a, b in params: print(f'\t{a}: {b}')

    print_title(f'{trial} trials {K}-fold cross validation')
    result = K_fold_performance(data, g, gn, trial)
    result.to_csv(out)
    print_title('final result')
    print(result.filter(regex='^avg').replace('nan±nan', '-'))

def K_fold_performance(data, g, gn, trial_num):
    tasks = [
        ['CPH', baselines.sksurv, False],
        ['RSF', baselines.sksurv, False],
        ['DeepSurv', baselines.pycox, False],
        ['Nnet-survival', baselines.pycox, False],
        ['PMF', baselines.pycox, False],
        ['DeepHit', baselines.pycox, False],
        ['MLP', baselines.pycox, False],
        # ['MLP_Multi', baselines.pycox, True],
        # ['MLP', baselines.coxbased, False],
        # [f'MLP_Surv_{INTERVALS}', baselines.noncox, False],

        [f'GCN_{gn}', baselines.pycox, True],
        [f'GAT_{gn}', baselines.pycox, True],
        # [f'GIN_{gn}', baselines.pycox, True],
        [f'SAG_{gn}', baselines.pycox, True],
        [f'MVPGCN_{gn}', baselines.pycox, True],
        [f'MVPGINGAT_{gn}', baselines.pycox, False],

        # [f'MVPGCN_Multi', baselines.pycox, True],
        # [f'MVPGINGAT_Multi', baselines.pycox, True],

        # [f'GCN_{gn}', baselines.coxbased, True],
        # [f'GAT_{gn}', baselines.coxbased, True],
        # [f'GIN_{gn}', baselines.coxbased, True],
        # [f'SAG_{gn}', baselines.coxbased, True],
        # [f'MVPGCN_{gn}', baselines.coxbased, True],
        # [f'MVPGINGAT_{gn}', baselines.coxbased, True],
        # [f'GIN_Surv_{INTERVALS}', baselines.noncox, True],
        # [f'MVP_Surv_{INTERVALS}', baselines.noncox, True],
    ]
    return run_K_fold(tasks, data, g, trial_num)

def run_K_fold(tasks, data, g, trial_num):
    kf = StratifiedKFold(n_splits=K, random_state=GLOBALSEED, shuffle=True)

    result = None
    for trial in tqdm.trange(trial_num):
        tqdm.tqdm.write(f'\nTrial {trial+1}:')
        for fold_k, (train_index, test_index) in enumerate(kf.split(data.index, data.event)):
            tqdm.tqdm.write(f'\nFold {fold_k+1} [train] {len(train_index)} [test] {len(test_index)}:')
            train = data.iloc[train_index]
            test = data.iloc[test_index]

            arr = []
            for name, func, need_graph, *params in tasks:
                params = params[0] if params else {}
                if need_graph and g is None: continue
                arr.append([name, *func(train, test, name, g, **params)])

            fold_result = pd.DataFrame(arr, columns=['method', f'T{trial+1}_F{fold_k+1}_train_C', f'T{trial+1}_F{fold_k+1}_test_C', f'T{trial+1}_F{fold_k+1}_test_Ctd', f'T{trial+1}_F{fold_k+1}_test_IBS'])
            fold_result.set_index('method', inplace=True)

            if result is None:
                result = fold_result
            else:
                result = result.merge(fold_result, left_index=True, right_index=True)

        result[f'T{trial+1}_avg_train_C'] = result.filter(regex=f'T{trial+1}_F\d_train_C$').to_numpy().mean(1)
        result[f'T{trial+1}_avg_test_C'] = result.filter(regex=f'T{trial+1}_F\d_test_C$').to_numpy().mean(1)
        result[f'T{trial+1}_avg_test_Ctd'] = result.filter(regex=f'T{trial+1}_F\d_test_Ctd').to_numpy().mean(1)
        result[f'T{trial+1}_avg_test_IBS'] = result.filter(regex=f'T{trial+1}_F\d_test_IBS').to_numpy().mean(1)


    train_c = result.filter(regex='T*_avg_train_C$') * 100
    test_c = result.filter(regex='T*_avg_test_C$') * 100
    test_ctd = result.filter(regex='T*_avg_test_Ctd') * 100
    test_IBS = result.filter(regex='T*_avg_test_IBS') * 100
    result['avg_train_C (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(train_c.mean(1), train_c.var(1))]
    result['avg_test_C (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(test_c.mean(1), test_c.var(1))]
    result['avg_test_Ctd (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(test_ctd.mean(1), test_ctd.var(1))]
    result['avg_test_IBS (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(test_IBS.mean(1), test_IBS.var(1))]

    return result

def grid_search(dataset, mode='cli_g85', trial=TRIAL):
    data, g, gn = set_mode(dataset, mode)

    lr = [0.001, 0.0005, 0.0001]
    readout = ['both', 'max', 'avg']
    fusion = ['cat', 'dot']
    dropout = [0.1, 0.3, 0.5]
    haz_dim = [128, 64, 32, 16]
    g_hid_layers = [2, 3, 4]
    g_hid_dim = [64, 32, 16]
    c_emb_dim = [64, 32, 16]

    print(f'params:')
    print(f'{lr = }')
    print(f'{readout = }')
    print(f'{fusion = }')
    print(f'{dropout = }')
    print(f'{haz_dim = }')
    print(f'{g_hid_layers = }')
    print(f'{g_hid_dim = }')
    print(f'{c_emb_dim = }')

    combinations = []
    for r in readout:
        for f in fusion:
            for e in g_hid_dim:
                for c in c_emb_dim:
                    if f == 'dot':
                        t = e * (2 if r == 'both' else 1)
                        if t != c: continue

                    for y in g_hid_layers:
                        for h in haz_dim:
                            for d in dropout:
                                for l in lr:
                                    config = {
                                        'lr': l,
                                        'net_param': {
                                            'readout': r,
                                            'fusion': f,
                                            'dropout': d,
                                            'c_embed_dim': c,
                                            'haz_dim': h,
                                            'g_hid_layers': y,
                                            'g_hid_dim': e,
                                        }
                                    }
                                    combinations.append(config)

    print(f'{len(combinations) = }')

    print_title(f'{trial} trials grid search')
    tasks = []
    for i, config in enumerate(combinations):
        tasks.append([f'GCN_{i:0>4d}', baselines.pycox, True, config])

    result = run_K_fold(tasks, data, g, trial)
    
    print_title('final result')
    sorted_r = result.filter(regex='^avg').replace('nan±nan', '-').sort_values(by=['avg_test_C (%)','avg_test_Ctd (%)'])
    print(sorted_r)

    for i in sorted_r[-10:].index:
        i = int(i[-4:])
        print(i, combinations[i])

def visualize_microbiome_graph_k_fold(data, graphs, gname='UnD'):
    global node_taxid, cnt
    model_dir = Path('../model')
    graph = graphs[gname]
    method = 'MVPGAT'

    if (model_dir / f'{method}_{gname}_r{MVPOOL_RATIO}_f1.pt').exists():
        test_loader = DataLoader(Dataset(data), batch_size=data.shape[0], shuffle=False)
        gn = graph.num_nodes()
        g2i = {g:i for i, g in enumerate(data.columns[-gn:].map(int))}
        inp_dim_n = 1
        inp_dim_g = data.shape[1] - 2 - gn
        hid_dim_n = 64
        hid_dim_g = 64

        ArgStruct = namedtuple('args', 'dropout_ratio hop hop_connection lamb nhid ginp ghid patience pooling_ratio sample_neighbor sparse_attention structure_learning num_features num_classes')
        args = ArgStruct(
            dropout_ratio=0.1,
            hop=3,
            hop_connection=False,
            lamb=2.0,
            nhid=hid_dim_n,
            ginp=inp_dim_g,
            ghid=hid_dim_g,
            patience=100,
            pooling_ratio=MVPOOL_RATIO,
            sample_neighbor=True,
            sparse_attention=True,
            structure_learning=False,
            num_features=inp_dim_n,
            num_classes=1
        )

        def plot_heatmap(mat, df, not_found, out, **kwargs):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10),
                                gridspec_kw={'width_ratios': [1, 30], 'wspace':-0.25},
                                sharey=True)

            colors = ['#9AEBA3', '#0FC2C0']
            df.event = df.event.astype(int)
            df.index.name = 'Patients'
            df['y'] = np.arange(len(df))+0.1
            for e in range(2):
                ax1.barh(df.y[df.event==e], df.time[df.event==e],
                        align='edge', color=colors[e], label=e)
            sns.despine(left=True, right=True)
            ax1.invert_xaxis()
            ax1.set_xticks([0,400])
            ax1.set_xticklabels([0,400], fontsize=5)
            ax1.set_xlabel('time', fontsize=5)
            ax1.legend(ncol=1, frameon=False, fontsize=5)

            mask = mat.numpy()==0
            sns.heatmap(mat, mask=mask, cmap="YlGnBu", square=True, 
                            cbar_kws={"shrink": .5}, linewidths=.5,
                            xticklabels=False, yticklabels=False, 
                            ax=ax2,
                            **kwargs)

            ticks, labels = zip(*[(g2i[taxid]+0.5, g) for taxid, g in KEY_GENUS.items()])
            ax2.set_xticks(ticks, labels, rotation=-90, fontsize=3)
            for g in not_found:
                ax2.get_xticklabels()[labels.index(g)].set_color('red')

            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

        def hook(perm, batch, score, df):
            global node_taxid, cnt
            cnt += 1
            m = batch[-1] + 1
            n = torch.div(batch.shape[0], m, rounding_mode='trunc')
            node_taxid = node_taxid.flatten()[perm].view(m,n)
            print(f'total: {node_taxid.shape}, unique node: {len(node_taxid.unique())}')
            
            if cnt == 3:
                not_found = [KEY_GENUS[g] for g in KEY_GENUS if g not in node_taxid.unique()]
                print(f'has {len(KEY_GENUS)-len(not_found)}/{len(KEY_GENUS)} key genus, not found:')
                print(not_found)
                print(node_taxid)
                score = score.view(m,n)
                img = f'../img/{method}_Und_r{MVPOOL_RATIO}_f{fold_i+1}.png'
                mat = torch.zeros(m, gn)
                for i in range(m):
                    for j in range(n):
                        mat[i, g2i[node_taxid[i, j].item()]] = score[i, j]
                plot_heatmap(mat, df, not_found, img)

        model = MVPoolGraphModel(args).to(DEVICE)
        for fold_i in range(K):
            node_taxid = torch.IntTensor([data.columns[-graph.num_nodes():].map(int)] * data.shape[0])
            cnt = 0

            print(f'Fold {fold_i+1}:')
            model.load_state_dict(torch.load(model_dir / f'{method}_Und_r{MVPOOL_RATIO}_f{fold_i+1}.pt', map_location=DEVICE))
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x, yevent, ytime, ind = map(lambda d: d.to(DEVICE), sort_data2(x, y))
                    inp_n = x[:, inp_dim_g:]
                    inp_n = inp_n.reshape(-1).unsqueeze(1)
                    inp_g = x[:, :inp_dim_g]
                    batch_g = dgl.batch([graph] * x.shape[0])
                    edge_index = torch.stack(batch_g.edges()).to(DEVICE)
                    which_graph = torch.tensor([[i] * graph.num_nodes() for i in range(x.shape[0])]).flatten().to(DEVICE)
                    
                    df = pd.DataFrame(torch.stack([yevent, ytime]).T, columns=['event', 'time'], index=data.index[ind])
                    hook_fn = partial(hook, df=df)
                    p = model(inp_n, inp_g, edge_index, which_graph, hook=hook_fn)
                    loss = neg_log_likelihood(p, ytime, yevent)
                    test_cindex = concordance_index(ytime, -p[:,0], yevent)

                    print(f'loss: {loss.item():.4f}, Ctd: {test_cindex:.4f}')

    else:
        kf = StratifiedKFold(n_splits=K, random_state=GLOBALSEED, shuffle=True)
        result = None
        print(f'Params: [Batch] {BATCH} [LR] {LR} [Epochs] {EPOCHS} [Dropout] {DROPOUT} [SAGPool_ratio] {SAGPOOL_RATIO} [K-fold] {K} [Trial] {TRIAL} [Device] {DEVICE}\n')

        for fold_k, (train_index, test_index) in enumerate(kf.split(data.index, data.event)):
            print(f'Fold {fold_k+1} [train] {len(train_index)} [test] {len(test_index)}:')
            train = data.iloc[train_index]
            test = data.iloc[test_index]

            fold_result = pd.DataFrame([
                # ['GAT_Und', *test_GNN(train, test, graphs['UnD'], 'GAT_UnD', save_model=f'../model/GAT_Und_f{fold_k+1}.pt')],
                [f'{method}_{gname}', *test_MVPool(train, test, graphs[gname], f'{method}_{gname}', save_model=f'../model/{method}_{gname}_r{MVPOOL_RATIO}_f{fold_k+1}.pt')],
            ] if graphs else [], columns=['method', f'F{fold_k+1}_train_cindex', f'F{fold_k+1}_test_cindex', f'F{fold_k+1}_test_IBS'])
            fold_result.set_index('method', inplace=True)

            if result is None:
                result = fold_result
            else:
                result = result.merge(fold_result, left_index=True, right_index=True)
            print()

        train_ctd = result.filter(regex='F\d_train_cindex') * 100
        test_ctd = result.filter(regex='F\d_test_cindex') * 100
        test_IBS = result.filter(regex='F\d_test_IBS') * 100
        result['avg_train_Ctd (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(train_ctd.mean(1), train_ctd.var(1))]
        result['avg_test_Ctd (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(test_ctd.mean(1), test_ctd.var(1))]
        result['avg_test_IBS (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(test_IBS.mean(1), test_IBS.var(1))]

        print(result.filter(regex='^avg').replace('nan±nan', '-'))

def visualize_microbiome_graph(data, graphs, gname='UnD'):
    global node_taxid, cnt
    model_dir = Path('../model')
    graph = graphs[gname]
    method = 'MVPGAT'

    if (model_dir / f'{method}_{gname}_r{MVPOOL_RATIO}_all.pt').exists():
        test_loader = DataLoader(Dataset(data), batch_size=data.shape[0], shuffle=False)
        gn = graph.num_nodes()
        g2i = {g:i for i, g in enumerate(graphs['ori_cols'].map(int))}
        inp_dim_n = 1
        inp_dim_g = data.shape[1] - 2 - gn
        hid_dim_n = 64
        hid_dim_g = 64

        ArgStruct = namedtuple('args', 'dropout_ratio hop hop_connection lamb nhid ginp ghid patience pooling_ratio sample_neighbor sparse_attention structure_learning num_features num_classes')
        args = ArgStruct(
            dropout_ratio=0.1,
            hop=3,
            hop_connection=False,
            lamb=2.0,
            nhid=hid_dim_n,
            ginp=inp_dim_g,
            ghid=hid_dim_g,
            patience=100,
            pooling_ratio=MVPOOL_RATIO,
            sample_neighbor=True,
            sparse_attention=True,
            structure_learning=False,
            num_features=inp_dim_n,
            num_classes=1
        )

        def plot_heatmap(mat, df, not_found, out, **kwargs):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10),
                                gridspec_kw={'width_ratios': [1, 30], 'wspace':-0.25},
                                sharey=True)

            colors = ['#9AEBA3', '#0FC2C0']
            df.event = df.event.astype(int)
            df.index.name = 'Patients'
            df['y'] = np.arange(len(df))+0.1
            for e in range(2):
                ax1.barh(df.y[df.event==e], df.time[df.event==e],
                        align='edge', color=colors[e], label=e)
            sns.despine(left=True, right=True)
            ax1.invert_xaxis()
            ax1.set_xticks([0,400])
            ax1.set_xticklabels([0,400], fontsize=5)
            ax1.set_xlabel('time', fontsize=5)
            ax1.legend(ncol=1, frameon=False, fontsize=5)

            mask = mat.numpy()==0
            # # min_max
            for i in range(mat.shape[0]):
                min = mat[i, ~mask[i]].min()
                max = mat[i, ~mask[i]].max()
                mat[i, ~mask[i]] = (mat[i, ~mask[i]] - min) / (max - min)
            
            # standard scala
            # for i in range(mat.shape[0]):
            #     u = mat[i, ~mask[i]].mean()
            #     std = mat[i, ~mask[i]].std()
            #     mat[i, ~mask[i]] = (mat[i, ~mask[i]] - u) / std
            
            sns.heatmap(mat, mask=mask, cmap="YlGnBu", square=True, 
                            cbar_kws={"shrink": .5}, linewidths=.5,
                            xticklabels=False, yticklabels=False, 
                            ax=ax2,
                            **kwargs)
            # breakpoint()
            ticks, labels = zip(*[(g2i[taxid]+0.5, g) for taxid, g in KEY_GENUS.items()])
            ax2.set_xticks(ticks, labels, rotation=-90, fontsize=3)
            for g in not_found:
                ax2.get_xticklabels()[labels.index(g)].set_color('red')

            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

            k = 30
            score, indices = mat.mean(0).topk(k)
            topk = graphs['ori_cols'][indices]
            topk_n = [TAXID_NAME[i] for i in topk]
            print(f'top {k}: {topk_n}')
            print(f'find: {set(topk_n) & set(KEY_GENUS.values())}')
            print(topk)

        def plot_heatmap_horizon(mat, df, not_found, out, **kwargs):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                gridspec_kw={'height_ratios': [1, 20], 'hspace':-0.1},
                                sharex=True)

            colors = ['#9AEBA3', '#0FC2C0']
            df.event = df.event.astype(int)
            df.index.name = 'Patients'
            df['x'] = np.arange(len(df), 0, -1)+0.1
            for e in range(2):
                ax1.bar(df.x[df.event==e], df.time[df.event==e],
                        align='edge', color=colors[e], label=e)
            sns.despine(right=True, top=True, bottom=True)
            ax1.set_yticks([0,400])
            ax1.set_yticklabels([0,400], fontsize=5)
            ax1.set_ylabel('time', fontsize=5, rotation=0)
            ax1.legend(ncol=1, frameon=False, fontsize=5)

            mask = mat.numpy()==0
            # # min_max
            for i in range(mat.shape[0]):
                min = mat[i, ~mask[i]].min()
                max = mat[i, ~mask[i]].max()
                mat[i, ~mask[i]] = (mat[i, ~mask[i]] - min) / (max - min)
            
            # standard scala
            # for i in range(mat.shape[0]):
            #     u = mat[i, ~mask[i]].mean()
            #     std = mat[i, ~mask[i]].std()
            #     mat[i, ~mask[i]] = (mat[i, ~mask[i]] - u) / std
            
            hm = sns.heatmap(mat.T, mask=mask.T, cmap="YlGnBu", square=True, 
                            cbar_kws={"shrink": .5}, linewidths=.5,
                            xticklabels=False, yticklabels=False, 
                            ax=ax2, cbar=False,
                            **kwargs)
            # breakpoint()
            ticks, labels = zip(*[(g2i[taxid]+0.5, g) for taxid, g in KEY_GENUS.items()])
            ax2.set_yticks(ticks, labels, fontsize=4)
            for g in not_found:
                ax2.get_yticklabels()[labels.index(g)].set_color('red')
            ax2.set_xlabel('samples', fontsize=8)

            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

            k = 30
            score, indices = mat.mean(0).topk(k)
            topk = graphs['ori_cols'][indices]
            topk_n = [TAXID_NAME[i] for i in topk]
            print(f'top {k}: {topk_n}')
            print(f'find: {set(topk_n) & set(KEY_GENUS.values())}')
            print(topk)


        def hook(perm, batch, score, df):
            global node_taxid, cnt
            cnt += 1
            m = batch[-1] + 1
            n = torch.div(batch.shape[0], m, rounding_mode='trunc')
            node_taxid = node_taxid.flatten()[perm].view(m,n)
            print(f'total: {node_taxid.shape}, unique node: {len(node_taxid.unique())}')
            
            if cnt == 3:
                not_found = [KEY_GENUS[g] for g in KEY_GENUS if g not in node_taxid.unique()]
                print(f'has {len(KEY_GENUS)-len(not_found)}/{len(KEY_GENUS)} key genus, not found:')
                print(not_found)
                print(node_taxid)
                score = score.view(m,n)
                img = f'../img/{method}_{gname}_r{MVPOOL_RATIO}_all_minmax_horizon.png'
                mat = torch.zeros(m, len(g2i))
                for i in range(m):
                    for j in range(n):
                        mat[i, g2i[node_taxid[i, j].item()]] = score[i, j]
                plot_heatmap_horizon(mat, df, not_found, img)


        node_taxid = torch.IntTensor([data.columns[-gn:].map(int)] * data.shape[0])
        cnt = 0
        model = MVPoolGATModel(args).to(DEVICE)
        model.load_state_dict(torch.load(model_dir / f'{method}_{gname}_r{MVPOOL_RATIO}_all.pt', map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, yevent, ytime, ind = map(lambda d: d.to(DEVICE), sort_data2(x, y))
                inp_n = x[:, inp_dim_g:]
                inp_n = inp_n.reshape(-1).unsqueeze(1)
                inp_g = x[:, :inp_dim_g]
                batch_g = dgl.batch([graph] * x.shape[0])
                edge_index = torch.stack(batch_g.edges()).to(DEVICE)
                which_graph = torch.tensor([[i] * graph.num_nodes() for i in range(x.shape[0])]).flatten().to(DEVICE)

                df = pd.DataFrame(torch.stack([yevent, ytime]).T, columns=['event', 'time'], index=data.index[ind])
                hook_fn = partial(hook, df=df)
                p = model(inp_n, inp_g, edge_index, which_graph, hook=hook_fn)

    else:
        test_MVPool(data, data, graphs[gname], f'{method}_{gname}', save_model=f'../model/{method}_{gname}_r{MVPOOL_RATIO}_all.pt')

def reduction_plot(x, hue_data, style_data, method='tsne'):
    data = pd.DataFrame([])

    if method == 'tsne':
        from sklearn.manifold import TSNE
        init = 'pca'
        # init = 'random'
        embedding = TSNE(n_components=2, init=init, learning_rate=200).fit_transform(x)
    elif method == 'umap':
        import umap
        init = 'spectral'
        # init = 'random'
        embedding = umap.UMAP(random_state=GLOBALSEED, init=init).fit_transform(x)

    out = f'{method}_{init}.png'
    print(f'plot the 2D reduction figure: {out}')

    data['x'] = embedding[:,0]
    data['y'] = embedding[:,1]
    data['OS time'] = hue_data
    data['cancer'] = style_data
    plt.rcParams['font.sans-serif'] = ['Source Han Sans JP']
    sns.scatterplot(data=data, x='x', y='y', hue='OS time', style='cancer', linewidth=0, s=20)
    plt.legend(fontsize=5)
    # plt.legend(loc='upper right', fontsize=7)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

def heatmap_plot(mat, df, not_found, out, **kwargs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 18),
                        gridspec_kw={'width_ratios': [1, 15], 'wspace':-0.25},
                        sharey=True)

    colors = ['#9AEBA3', '#0FC2C0']
    df.event = df.event.astype(int)
    df.index.name = 'Patients'
    df['y'] = np.arange(len(df))+0.1
    for e in range(2):
        ax1.barh(df.y[df.event==e], df.time[df.event==e],
                align='edge', color=colors[e], label=e)
    sns.despine(left=True, right=True)
    ax1.invert_xaxis()
    ax1.set_xticks([0,850])
    ax1.set_xticklabels([0,850], fontsize=5)
    ax1.set_xlabel('time', fontsize=5)
    ax1.legend(ncol=1, frameon=False, fontsize=5)

    mask = mat==0
    # min_max
    for i in range(mat.shape[0]):
        min = mat[i, ~mask[i]].min()
        max = mat[i, ~mask[i]].max()
        mat[i, ~mask[i]] = (mat[i, ~mask[i]] - min) / (max - min)
    
    # standard scala
    # for i in range(mat.shape[0]):
    #     u = mat[i, ~mask[i]].mean()
    #     std = mat[i, ~mask[i]].std()
    #     mat[i, ~mask[i]] = (mat[i, ~mask[i]] - u) / std

    sns.heatmap(mat, mask=mask, cmap="YlGnBu", square=True, 
                    cbar_kws={"shrink": .5}, linewidths=.1,
                    xticklabels=False, yticklabels=False, 
                    ax=ax2,
                    **kwargs)
    # breakpoint()
    # ticks, labels = zip(*[(g2i[taxid]+0.5, g) for taxid, g in KEY_GENUS.items()])
    # ax2.set_xticks(ticks, labels, rotation=-90, fontsize=3)
    # for g in not_found:
    #     ax2.get_xticklabels()[labels.index(g)].set_color('red')

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

def visualization(method='GCN', mode='cli_g85'):
    print_title(f'visualization of {method}')

    model_dir = Path('../model')
    model_pt = model_dir / f'vis_{method}_all_{mode}.pt'
    data, g, gn = set_mode(dataset, mode)
    g2i = {g:i for i, g in enumerate(data.columns[-g.num_nodes():].map(int))}

    # mat = np.load('mat.npy', allow_pickle=True)
    # heatmap_plot(mat, data[['event', 'time']], [], '1.png')
    # return

    print(f'model file: {model_pt}')
    if model_pt.exists():
        model, fusion = baselines.pycox(data, data, f'{method}_{gn}', g, training=False, save_model=str(model_pt.resolve()), return_fusion=True)
    else:
        print('training model ...')
        model, fusion = baselines.pycox(data, data, f'{method}_{gn}', g, save_model=str(model_pt.resolve()), return_fusion=True)

    cancer = pd.DataFrame(['未知']*len(data), index=data.index)
    for c in data.columns[11:29]: cancer[data[c]==1] = c
    # reduction_plot(fusion.detach().numpy(), data.time.values, cancer[0].values, 'tsne')
    # reduction_plot(fusion.detach().numpy(), data.time.values, cancer[0].values, 'umap')

    # =================
    # visualizing node
    # =================
    global node_taxid, cnt
    node_taxid = torch.IntTensor([data.columns[-g.num_nodes():].map(int)] * data.shape[0])
    cnt = 0

    def hook(perm, batch, score, df=None):
        global node_taxid, cnt
        cnt += 1
        m = batch[-1] + 1
        n = torch.div(batch.shape[0], m, rounding_mode='trunc')
        node_taxid = node_taxid.flatten()[perm].view(m,n)
        print(f'total: {node_taxid.shape}, unique node: {len(node_taxid.unique())}')

        if cnt == 3:
            not_found = [KEY_GENUS[g] for g in KEY_GENUS if g not in node_taxid.unique()]
            print(f'has {len(KEY_GENUS)-len(not_found)}/{len(KEY_GENUS)} key genus, not found:')
            print(not_found)
            print(node_taxid)
            score = score.view(m,n)

            img = f'{method}_{mode}_r{MVPOOL_RATIO}_all_minmax.png'
            mat = torch.zeros(m, len(g2i))
            for i in range(m):
                for j in range(n):
                    mat[i, g2i[node_taxid[i, j].item()]] = score[i, j]

            nonzero_cols = []
            for j in range(len(g2i)):
                if mat[:,j].sum() != 0:
                    nonzero_cols.append(j)

            heatmap_plot(mat[:,nonzero_cols].detach().numpy(), df, not_found, img)

            k = 10
            _, indices = mat.mean(0).topk(k)
            topk = data.columns[-g.num_nodes():][indices].map(str)
            topk_n = [TAXID_NAME[i] for i in topk]
            print(f'top {k}: {topk_n}')
            print(f'find: {set(topk_n) & set(KEY_GENUS.values())}')
            print(f'find(mm): {set(topk_n) & set(MagMela_GENUS.values())}')
            print(topk)

    test = torch.tensor(data.iloc[:,2:].values.astype(np.float32)).to(DEVICE)
    hook_fn = partial(hook, df=data[['event', 'time']])
    model.net(test, hook=hook_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='cli_g85', help='run different testing mode')
    parser.add_argument('-t', '--trial', type=int, default=TRIAL, help='run n trials k-fold cross validation')
    parser.add_argument('-v', '--visualize', default=False, action='store_true', help='for visualization')
    parser.add_argument('-g', '--gridsearch', default=False, action='store_true', help='for grid search')
    args = parser.parse_args()

    watermark()

    if args.visualize:
        visualization('MVPGINGAT', args.mode)
    elif args.gridsearch:
        grid_search(dataset, args.mode, args.trial)
    else:
        run_benchmark(dataset, args.mode, args.trial)
