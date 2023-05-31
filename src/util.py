import tqdm
import baselines
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from sklearn.model_selection import StratifiedKFold

def get_func(model_name):
    if model_name in ['CPH', 'RSF']:
        return baselines.sksurv
    elif model_name in ['AE']:
        return baselines.coxbased
    else:
        return baselines.pycox

def run_K_fold(K, task_list, trial_num):
    is_dataset_same_size = len(set(len(t[2]) for t in task_list)) == 1
    if is_dataset_same_size:
        result, (record, record2) = _run_by_fold(K, trial_num, task_list)
    else:
        result, (record, record2) = _run_by_task(K, trial_num, task_list)

    train_c = result.filter(regex='T*_avg_train_C$') * 100
    test_c = result.filter(regex='T*_avg_test_C$') * 100
    test_ctd = result.filter(regex='T*_avg_test_Ctd') * 100
    test_IBS = result.filter(regex='T*_avg_test_IBS') * 100
    result['avg_train_C (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(train_c.mean(1), train_c.var(1))]
    result['avg_test_C (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(test_c.mean(1), test_c.var(1))]
    result['avg_test_Ctd (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(test_ctd.mean(1), test_ctd.var(1))]
    result['avg_test_IBS (%)'] = [f'{a:.2f}±{b:.2f}' for a, b in zip(test_IBS.mean(1), test_IBS.var(1))]

    return result, (record, record2)

def _run_by_task(K, trial_num, task_list):
    result = None
    record2 = None

    record_d = {}
    for name, model_name, data, _ in task_list:
        r = data[['event','time']]
        r[f'{name}_p'] = r[f'{name}_c'] = None
        record_d[name] = r

    for trial in tqdm.trange(trial_num):
        tqdm.tqdm.write(f'\nTrial {trial+1}:')

        trial_arr = []
        trial_cols = ['method']
        for fold_k in range(K):
            trial_cols += [f'T{trial+1}_F{fold_k+1}_train_C', f'T{trial+1}_F{fold_k+1}_test_C', f'T{trial+1}_F{fold_k+1}_test_Ctd', f'T{trial+1}_F{fold_k+1}_test_IBS']
        
        for name, model_name, data, g in task_list:
            tqdm.tqdm.write(f'\nTask: {name}, model: {model_name}')

            kf = StratifiedKFold(n_splits=K, random_state=np.random.randint(0, 10000), shuffle=True)
            fold_cols = ['method']
            fold_arr = [name]
            for fold_k, (train_index, test_index) in enumerate(kf.split(data.index, data.event)):
                tqdm.tqdm.write(f'\nFold {fold_k+1} [train] {len(train_index)} [test] {len(test_index)}:')

                train = data.iloc[train_index]
                test = data.iloc[test_index]
                func = get_func(model_name)
                *r, pred_test = func(train, test, model_name, g)

                record_d[name].iloc[test_index, record_d[name].columns.to_list().index(f'{name}_p')] = pred_test
                record_d[name].iloc[test_index, record_d[name].columns.to_list().index(f'{name}_c')] = (pred_test > np.median(pred_test)).astype(int)

                a = sorted(zip(pred_test, record_d[name].iloc[test_index].time))
                if len(a) % 2:
                    m_pfs = a[len(a)//2][1]
                else:
                    m_pfs = (a[len(a)//2-1][1] + a[len(a)//2][1]) / 2
                # tmp.append([name, np.median(pred_test), m_pfs])

                fold_arr += [*r]

            trial_arr.append(fold_arr)

        fold_result = pd.DataFrame(trial_arr, columns=trial_cols)
        fold_result.set_index('method', inplace=True)
        if result is None:
            result = fold_result
        else:
            result = result.merge(fold_result, left_index=True, right_index=True)

        result[f'T{trial+1}_avg_train_C'] = result.filter(regex=f'T{trial+1}_F\d_train_C$').to_numpy().mean(1)
        result[f'T{trial+1}_avg_test_C'] = result.filter(regex=f'T{trial+1}_F\d_test_C$').to_numpy().mean(1)
        result[f'T{trial+1}_avg_test_Ctd'] = result.filter(regex=f'T{trial+1}_F\d_test_Ctd').to_numpy().mean(1)
        result[f'T{trial+1}_avg_test_IBS'] = result.filter(regex=f'T{trial+1}_F\d_test_IBS').to_numpy().mean(1)

    return result, (record_d, None)

def _run_by_fold(K, trial_num, task_list):
    result = None
    record2 = None
    record = task_list[0][2][['event','time']]
    for name, *_ in task_list:
        record[f'{name}_p'] = record[f'{name}_c'] = None

    for trial in tqdm.trange(trial_num):
        tqdm.tqdm.write(f'\nTrial {trial+1}:')

        kf = StratifiedKFold(n_splits=K, random_state=np.random.randint(0, 10000), shuffle=True)
        for fold_k, (train_index, test_index) in enumerate(kf.split(record.index, record.event)):
            tqdm.tqdm.write(f'\nFold {fold_k+1} [train] {len(train_index)} [test] {len(test_index)}:')

            arr = []
            tmp = []
            for name, model_name, data, g in task_list:
                train = data.iloc[train_index]
                test = data.iloc[test_index]
                func = get_func(model_name)
                *r, pred_test = func(train, test, model_name, g)

                record.iloc[test_index, record.columns.to_list().index(f'{name}_p')] = pred_test
                record.iloc[test_index, record.columns.to_list().index(f'{name}_c')] = (pred_test > np.median(pred_test)).astype(int)
                arr.append([name, *r])

                a = sorted(zip(pred_test, record.iloc[test_index].time))
                if len(a) % 2:
                    m_pfs = a[len(a)//2][1]
                else:
                    m_pfs = (a[len(a)//2-1][1] + a[len(a)//2][1]) / 2
                tmp.append([name, np.median(pred_test), m_pfs])

            cols = ['method', f'T{trial+1}_F{fold_k+1}_train_C', f'T{trial+1}_F{fold_k+1}_test_C', f'T{trial+1}_F{fold_k+1}_test_Ctd', f'T{trial+1}_F{fold_k+1}_test_IBS']
            fold_result = pd.DataFrame(arr, columns=cols)
            fold_result.set_index('method', inplace=True)
            tmp_result = pd.DataFrame(tmp, columns=['method', f'F{fold_k+1}_M', f'F{fold_k+1}_T'])
            tmp_result.set_index('method', inplace=True)

            if result is None:
                result = fold_result
            else:
                result = result.merge(fold_result, left_index=True, right_index=True)

            if record2 is None:
                record2 = tmp_result
            else:
                record2 = record2.merge(tmp_result, left_index=True, right_index=True)

        result[f'T{trial+1}_avg_train_C'] = result.filter(regex=f'T{trial+1}_F\d_train_C$').to_numpy().mean(1)
        result[f'T{trial+1}_avg_test_C'] = result.filter(regex=f'T{trial+1}_F\d_test_C$').to_numpy().mean(1)
        result[f'T{trial+1}_avg_test_Ctd'] = result.filter(regex=f'T{trial+1}_F\d_test_Ctd').to_numpy().mean(1)
        result[f'T{trial+1}_avg_test_IBS'] = result.filter(regex=f'T{trial+1}_F\d_test_IBS').to_numpy().mean(1)
    
    return result, (record, record2)
