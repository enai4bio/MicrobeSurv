from sched import scheduler
import dgl
import tqdm
import dataset
import numpy as np
import pandas as pd
from config import *
import torchtuples as tt
from datetime import datetime
from pycox.evaluation import EvalSurv
from torchtuples.callbacks import Callback
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
from pycox.models import LogisticHazard, PMF, CoxPH, DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from models import MLP, CoxModel, NonCoxModel, EarlyStop, GNN, MVP, DeepSurv, Nnetsurvival

def eval_surv(surv_df, y_df):
    time_grid = np.linspace(y_df.time.min(), y_df.time.max(), IBS_GRID_N)
    ev = EvalSurv(surv_df, y_df.time.values, y_df.event.values, censor_surv='km')
    Ctd = ev.concordance_td()
    IBS = ev.integrated_brier_score(time_grid)
    return Ctd, IBS

def sksurv(train, test, method_name, graph=None, print=tqdm.tqdm.write):
    start_time = datetime.now()
    label_type = [('cens', '?'), ('time', '<f8')]
    y_train = np.array([(bool(i), d) for i, d in train.iloc[:,:2].to_numpy()], dtype=label_type)
    y_test = np.array([(bool(i), d) for i, d in test.iloc[:,:2].to_numpy()], dtype=label_type)

    X_train = train.iloc[:,2:].to_numpy()
    X_test = test.iloc[:,2:].to_numpy()

    if method_name == 'CPH':
        model = CoxPHSurvivalAnalysis(alpha=CPH_PENALTY)
    elif method_name == 'RSF':
        model = RandomSurvivalForest(
                n_estimators=RSF_N_ESTIMATORS,
                min_samples_split=10,
                min_samples_leaf=15,
                max_features="sqrt",
                n_jobs=-1,
                random_state=GLOBALSEED)

    model.fit(X_train, y_train)
    # == sksurv.metrics.concordance_index_censored == lifelines.utils.concordance_index
    # >= sksurv.metrics.concordance_index_ipcw
    train_C = model.score(X_train, y_train)
    test_C = model.score(X_test, y_test)

    test_surv_fn = model.predict_survival_function(X_test)
    test_surv_df = pd.DataFrame(np.stack([fn.y for fn in test_surv_fn],axis=1), index=test_surv_fn[0].x)
    test_Ctd, test_IBS = eval_surv(test_surv_df, test)

    print(f'[{method_name}] [train] C: {train_C:.4f} | [test] C: {test_C:.4f}, Ctd: {test_Ctd:.4f}, IBS: {test_IBS:.4f} | time: {datetime.now() - start_time}')
    return train_C, test_C, test_Ctd, test_IBS

class LRScheduler(Callback):
    def __init__(self, lr, lr_decay=0.01):
        self.epoch = 0
        self.lr = lr
        self.lr_decay = lr_decay
        self.scheduler = scheduler

    def on_epoch_end(self):
        for p in self.model.optimizer.param_groups:
            p['lr'] = self.lr / (1 + self.epoch * self.lr_decay)

        self.epoch += 1

def pycox(train, test, method_name, graph=None, lr=LR, net_param=None, 
          training=True,
          save_model=None,
          return_hg=False,
          return_fusion=False,
          print=tqdm.tqdm.write):
    if method_name == 'Nnet-survival':
        method = LogisticHazard
    elif method_name == 'PMF':
        method = PMF
    elif method_name == 'DeepSurv':
        method = CoxPH
    elif method_name == 'DeepHit':
        method = DeepHitSingle
    elif method_name.endswith('Multi'):
        method = DeepHitSingle
    else:
        method = CoxPH

    start_time = datetime.now()
    valid = train.sample(frac=0.1)
    train = train.drop(valid.index)

    x_train = train.iloc[:,2:].values.astype(np.float32)
    y_train = (train.time.values.astype(np.float32), train.event.values.astype(np.float32))
    x_val = valid.iloc[:,2:].values.astype(np.float32)
    y_val = (valid.time.values.astype(np.float32), valid.event.values.astype(np.float32))
    x_test = test.iloc[:,2:].values.astype(np.float32)

    params = {}
    if method is not CoxPH:
        labtrans = LogisticHazard.label_transform(INTERVALS)
        y_train = labtrans.fit_transform(*y_train)
        y_val = labtrans.transform(*y_val)
        params = {
            'duration_index': labtrans.cuts
        }

    inp_dim = x_train.shape[1]
    L2_reg = 0

    callbacks = [tt.cb.EarlyStopping()]

    if method_name[:3] in ['GCN', 'GAT', 'GIN', 'SAG']:
        if net_param:
            g_dims = [G_INP_DIM] + [net_param['g_hid_dim']] * net_param['g_hid_layers']
            c_embed_dim = net_param['c_embed_dim']
            haz_dim = net_param['haz_dim']
            dropout = net_param['dropout']
            readout = net_param['readout']
            fusion = net_param['fusion']
        else:
            g_dims = [G_INP_DIM] + [G_HID_DIM] * G_HID_LAYER
            c_embed_dim = EMB_DIM
            haz_dim = HAZ_DIM
            dropout = DROPOUT
            readout = G_READOUT
            fusion = G_FUSION
        out_dim = 1 if method is CoxPH else INTERVALS
        net = GNN(method_name[:3], inp_dim, graph.to(DEVICE), g_dims, c_embed_dim, haz_dim, out_dim, dropout, readout=readout, fusion=fusion)
    elif method_name.startswith('MVP'):
        g_dims = [G_INP_DIM] + [G_HID_DIM] * G_HID_LAYER
        c_embed_dim = EMB_DIM
        haz_dim = HAZ_DIM
        readout = G_READOUT
        fusion = G_FUSION
        out_dim = 1 if method is CoxPH else INTERVALS
        g = graph.to(DEVICE) if graph else None
        net = MVP(method_name[3:6], inp_dim, g, g_dims, c_embed_dim, haz_dim, out_dim, DROPOUT, MVPOOL_RATIO, readout=readout, fusion=fusion)
    elif method_name == 'DeepSurv':
        # METABRIC
        # lr = 0.01
        # L2_reg = 10.891
        # lr_decay = 4.169e-3
        # hyperparams = {
        #     'batch_norm': False,
        #     'dropout': 0.16,
        #     'hidden_layers_sizes': [41],
        #     'n_in': inp_dim,
        #     'activation': 'selu'
        # }

        # SUPPORT
        lr = 0.047
        L2_reg = 8.12
        lr_decay = 2.573e-3
        hyperparams = {
            'batch_norm': False,
            'dropout': 0.255,
            'hidden_layers_sizes': [44],
            'n_in': inp_dim,
            'activation': 'selu'
        }

        callbacks += [LRScheduler(lr, lr_decay)]
        net = DeepSurv(**hyperparams)
    elif method_name == 'Nnet-survival':
        # SUPPORT
        hid_dims = [7]
        out_dim = 1 if method is CoxPH else INTERVALS
        L2_reg = 0.1

        net = Nnetsurvival(inp_dim, out_dim, hid_dims)
    else:
        inp_g = graph.num_nodes() if graph else 0
        inp_c = inp_dim - inp_g
        out_dim = 1 if method is CoxPH else INTERVALS
        net = MLP(inp_c, inp_g, EMB_DIM, G_HID_DIM, G_HID_LAYER, HAZ_DIM, out_dim, DROPOUT)

    opt = tt.optim.Adam(lr, weight_decay=L2_reg)
    model = method(net, opt, device=DEVICE, **params)

    if training:
        log = model.fit(x_train, y_train, BATCH, EPOCHS, callbacks, val_data=(x_val, y_val), verbose=False)
        log_df = log.to_pandas()
        if method is CoxPH: model.compute_baseline_hazards()
        # if save_model: model.save_net(save_model)
        if save_model:
            torch.save(model.net.state_dict(), save_model)
            if hasattr(model, 'baseline_hazards_'):
                model.baseline_hazards_.to_pickle(save_model.split('.')[0]+'_blh.pickle')
    else:
        log_df = pd.DataFrame([0], columns=['train_loss'])
        # model.load_net(save_model, map_location=DEVICE) # this will go wrong
        model.net.load_state_dict(torch.load(save_model, map_location=DEVICE))
        blh_path = save_model.split('.')[0]+'_blh.pickle'
        if os.path.isfile(blh_path):
            model.baseline_hazards_ = pd.read_pickle(blh_path)
            model.baseline_cumulative_hazards_ = model.baseline_hazards_.cumsum()

    train_C = test_C = pd.NA
    if method is CoxPH:
        train_pred = model.predict(x_train)
        train_C = concordance_index_censored(train.event.astype(bool), train.time, train_pred.squeeze())[0]
        test_pred = model.predict(x_test)
        test_C = concordance_index_censored(test.event.astype(bool), test.time, test_pred.squeeze())[0]

    test_surv_df = model.predict_surv_df(x_test)
    test_Ctd, test_IBS = eval_surv(test_surv_df, test)

    print(f'[{method_name}] [train] epoch: {len(log_df)}, loss: {log_df.train_loss.iloc[-1]:.4f}, C: {train_C:.4f} | [test] C: {test_C:.4f}, Ctd: {test_Ctd:.4f}, IBS: {test_IBS:.4f} | time: {datetime.now() - start_time}')

    if return_hg:
        return model, model.net(torch.tensor(x_test).to(DEVICE), return_hg=return_hg).cpu()
    if return_fusion:
        return model, model.net(torch.tensor(x_test).to(DEVICE), return_fusion=return_fusion).cpu()
    
    return train_C, test_C, test_Ctd, test_IBS

def gen_data_for_graph_model(x, dim, g, is_mvp=False):
    inp_c = x[:, :dim]
    inp_n = x[:, dim:]
    inp_n = inp_n.reshape(-1).unsqueeze(1)
    inp_g = dgl.batch([g] * x.shape[0])
    if is_mvp:
        edge_index = torch.stack(inp_g.edges()).to(DEVICE)
        which_graph = torch.tensor([[i] * (x.shape[1] - dim) for i in range(x.shape[0])]).flatten().to(DEVICE)
        return (inp_n, inp_c, edge_index, which_graph)
    return (inp_c, inp_n, inp_g.to(DEVICE))

def coxbased(train, test,
            method_name='MLP',
            graph=None,
            training=True,
            save_model=None,
            return_hg=False,
            verbose=False,
            print=tqdm.tqdm.write):
    start_time = datetime.now()

    inp_dim = train.shape[1] - 2
    if graph:
        inp_dim -= graph.num_nodes()
        is_mvp = method_name.startswith('MVP')

    if method_name.startswith('MLP'):
        net = MLP(inp_dim, EMB_DIM, HAZ_DIM, OUT_DIM, DROPOUT).to(DEVICE)
    elif method_name.startswith('GCN'):
        net = GNN('GCN', G_INP_DIM, G_HID_DIM, inp_dim, EMB_DIM, HAZ_DIM, OUT_DIM, DROPOUT).to(DEVICE)
    elif method_name.startswith('GAT'):
        net = GNN('GAT', G_INP_DIM, G_HID_DIM, inp_dim, EMB_DIM, HAZ_DIM, OUT_DIM, DROPOUT, num_heads=NUM_HEADS).to(DEVICE)
    elif method_name.startswith('GIN'):
        net = GNN('GIN', G_INP_DIM, G_HID_DIM, inp_dim, EMB_DIM, HAZ_DIM, OUT_DIM, DROPOUT).to(DEVICE)
    elif method_name.startswith('SAG'):
        net = GNN('SAG', G_INP_DIM, G_HID_DIM, inp_dim, EMB_DIM, HAZ_DIM, OUT_DIM, DROPOUT, pool_ratio=SAGPOOL_RATIO).to(DEVICE)
    elif method_name.startswith('MVPGCN'):
        net = MVP('GCN', G_INP_DIM, G_HID_DIM, inp_dim, EMB_DIM, HAZ_DIM, OUT_DIM, DROPOUT, MVPOOL_RATIO).to(DEVICE)
    elif method_name.startswith('MVPGINGAT'):
        net = MVP('GIN', G_INP_DIM, G_HID_DIM, inp_dim, EMB_DIM, HAZ_DIM, OUT_DIM, DROPOUT, MVPOOL_RATIO).to(DEVICE)
    elif method_name.startswith('TEST'):
        net = TEST('GCN', G_INP_DIM, G_HID_DIM, inp_dim, EMB_DIM, HAZ_DIM, OUT_DIM, DROPOUT, MVPOOL_RATIO,
                    readout='avg', fusion='dot').to(DEVICE)
    
    model = CoxModel(net, LR)

    df2t = lambda df: map(lambda d: torch.FloatTensor(d.to_numpy()).to(DEVICE), [df.iloc[:,2:], df.time, df.event])
    train_log = ''
    if training:
        valid = train.sample(frac=0.1)
        train = train.drop(valid.index)
        train_loader = dataset.loader(train)

        min_valid_loss = np.inf
        best_net_param = None
        scheduler = EarlyStop()

        for e in range(EPOCHS):
            losses = []
            model.net.train()
            for batch in train_loader:
                x, yt, ye = map(lambda d: d.to(DEVICE), batch)
                if graph: x = gen_data_for_graph_model(x, inp_dim, graph, is_mvp)
                loss = model.update(x, yt, ye)
                if loss is not None: losses.append(loss)

            model.net.eval()
            with torch.no_grad():
                x, yt, ye = df2t(valid)
                if graph: x = gen_data_for_graph_model(x, inp_dim, graph, is_mvp)
                loss = model.update(x, yt, ye, False)

                if loss < min_valid_loss:
                    min_valid_loss = loss
                    best_net_param = model.net.state_dict()
                
                if scheduler.add(loss): break

            if verbose:
                print(f'[{method_name}] epoch {e+1:2d}, loss: [train] {np.mean(losses):.4f}, [valid] {loss:.4f}, [best] {min_valid_loss:.4f}, cnt: {scheduler.cnt}')

        train_log = f'epoch: {e+1:2d}, loss: {np.mean(losses):.4f}, '
        model.net.load_state_dict(best_net_param)
        if save_model: torch.save(model.net.state_dict(), save_model)
    else:
        model.net.load_state_dict(torch.load(save_model, map_location=DEVICE))
    
    model.net.eval()
    with torch.no_grad():
        x_train = torch.FloatTensor(train.iloc[:,2:].to_numpy()).to(DEVICE)
        x_test = torch.FloatTensor(test.iloc[:,2:].to_numpy()).to(DEVICE)
        if graph:
            inps_train = gen_data_for_graph_model(x_train, inp_dim, graph, is_mvp)
            inps_test = gen_data_for_graph_model(x_test, inp_dim, graph, is_mvp)
        else:
            inps_train = (x_train,)
            inps_test = (x_test,)
        model.compute_haz(inps_train, train[['time','event']])

        train_pred = model.net(*inps_train).cpu()
        train_C = concordance_index_censored(train.event.astype(bool), train.time, train_pred.squeeze())[0]
        test_pred = model.net(*inps_test).cpu()
        test_C = concordance_index_censored(test.event.astype(bool), test.time, test_pred.squeeze())[0]
        test_surv_df = model.pred_surv(inps_test)
        test_Ctd, test_IBS = eval_surv(test_surv_df, test)

        print(f'[{method_name}] [train] {train_log}C: {train_C:.4f} | [test] C: {test_C:.4f}, Ctd: {test_Ctd:.4f}, IBS: {test_IBS:.4f} | time: {datetime.now() - start_time}')

        if return_hg:
            return model.net(*inps_test, return_hg=return_hg).cpu()

    return train_C, test_C, test_Ctd, test_IBS

def noncox(train, test, method_name='MLP', graph=None, save_model=None, verbose=False, print=tqdm.tqdm.write):
    start_time = datetime.now()

    valid = train.sample(frac=0.1)
    train = train.drop(valid.index)

    labtrans = LabTransDiscreteTime(INTERVALS)
    yt_train, _ = labtrans.fit_transform(train.time, train.event)
    yt_val, _ = labtrans.transform(valid.time, valid.event)
    valid.time = (yt_val - valid.event + 1).apply(lambda x: [1]*x+[0]*(INTERVALS-x))
    train_loader = dataset.loader(train, yt_train)
    
    # visulaze discrete time intervals
    # k=20
    # print('time:     '+' '.join(map(lambda x:f'{int(x):3d}', train.time.values[:k])))
    # print('interval: '+' '.join(map(lambda x:f'{int(x):3d}', yt_train[:k])))
    # print('event:    '+' '.join(map(lambda x:f'{int(x):3d}', train.event.values[:k])))
    # print('cuts: '+' '.join(map(lambda x:f'{x:.1f}', labtrans.cuts)))

    inp_dim = train.shape[1] - 2
    if graph:
        inp_dim -= graph.num_nodes()
        is_mvp = method_name.startswith('MVP')

    if method_name.startswith('MLP'):
        net = MLP(inp_dim, EMB_DIM, HAZ_DIM, INTERVALS, DROPOUT).to(DEVICE)
    elif method_name.startswith('GIN'):
        net = GNN('GIN', G_INP_DIM, G_HID_DIM, inp_dim, EMB_DIM, HAZ_DIM, INTERVALS, DROPOUT).to(DEVICE)
    elif method_name.startswith('MVP'):
        net = MVP('GIN', G_INP_DIM, G_HID_DIM, inp_dim, EMB_DIM, HAZ_DIM, INTERVALS, DROPOUT, MVPOOL_RATIO).to(DEVICE)
    model = NonCoxModel(net, LR, labtrans.cuts)

    df2t = lambda df: map(lambda d: torch.FloatTensor(d).to(DEVICE), [df.iloc[:,2:].to_numpy(), df.time.to_list(), df.event.to_numpy()])
    min_valid_loss = np.inf
    best_net_param = None
    scheduler = EarlyStop()

    for e in range(EPOCHS):
        losses = []
        model.net.train()
        for batch in train_loader:
            x, yt, ye = map(lambda d: d.to(DEVICE), batch)
            if graph: x = gen_data_for_graph_model(x, inp_dim, graph, is_mvp)
            loss = model.update(x, yt, ye)
            if loss is not None: losses.append(loss)
            

        model.net.eval()
        with torch.no_grad():
            x, yt, ye = df2t(valid)
            if graph: x = gen_data_for_graph_model(x, inp_dim, graph, is_mvp)
            loss = model.update(x, yt, ye, False)

            if loss < min_valid_loss:
                min_valid_loss = loss
                best_net_param = model.net.state_dict()
            
            if scheduler.add(loss): break

        if verbose:
            print(f'[{method_name}] epoch {e+1:2d}, loss: [train] {np.mean(losses):.4f}, [valid] {loss:.4f}, [best] {min_valid_loss:.4f}, cnt: {scheduler.cnt}')
    
    train_C = test_C = pd.NA
    model.net.load_state_dict(best_net_param)
    model.net.eval()
    with torch.no_grad():
        x_test = torch.FloatTensor(test.iloc[:,2:].to_numpy()).to(DEVICE)
        if graph:
            inps_test = gen_data_for_graph_model(x_test, inp_dim, graph, is_mvp)
        else:
            inps_test = (x_test,)

        test_surv_df = model.pred_surv(inps_test)
        test_Ctd, test_IBS = eval_surv(test_surv_df, test)

    print(f'[{method_name}] [train] epoch: {e+1:2d}, loss: {np.mean(losses):.4f}, C: {train_C:.4f} | [test] C: {test_C:.4f}, Ctd: {test_Ctd:.4f}, IBS: {test_IBS:.4f} | time: {datetime.now() - start_time}')
    if save_model: torch.save(model.net.state_dict(), save_model)

    return train_C, test_C, test_Ctd, test_IBS
