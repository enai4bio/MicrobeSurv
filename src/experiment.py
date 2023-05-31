from config import *
set_metaverse_seed(GLOBALSEED)
import dataset
from util import run_K_fold

def experiment1():
    '''
    头颈癌的对比实验
    '''
    g = dataset.g202_und
    hc_all = dataset.df_PFS_head_neck
    hc_cli_only = hc_all.iloc[:,:17]
    hc_microbe_only = hc_all[hc_all.columns[:2].tolist()+hc_all.columns[17:].tolist()]

    task_list = [
        # task_name, model, data, graph
        ['all', 'MicrobeSurv', hc_all, g],
        ['cli', 'MicrobeSurv', hc_cli_only, None],
        ['microbe', 'MicrobeSurv', hc_microbe_only, g]
    ]
    trial_num = 100
    result, *_ = run_K_fold(K, task_list, trial_num)
    print(result.filter(regex='^avg').replace('nan±nan', '-'))

def experiment2():
    '''
    固定PD实验
    '''
    g = dataset.g202_und
    all = dataset.df_et_PFS_cli_g202_b3
    
    PD_1 = all[all.PD == 1].drop(columns='PD') # 210, effect=0
    PD_0 = all[all.PD == 0].drop(columns='PD') # 293, effect=0,1
    
    task_list = [
        # task_name, model, data, graph
        ['all', 'MicrobeSurv', all, g],
        ['PD_1', 'MicrobeSurv', PD_1, g],
        ['PD_0', 'MicrobeSurv', PD_0, g]
    ]
    trial_num = 100
    result, *_ = run_K_fold(K, task_list, trial_num)
    print(result.filter(regex='^avg').replace('nan±nan', '-'))

if __name__ == '__main__':
    experiment2()
