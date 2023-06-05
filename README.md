# MicrobeSurv

Microbes have been linked to approximately 10–20% of human cancers, which can be the good resources for prognosis and survival prediction. Here we propose MicrobeSurv, a deep-learning based graph neural network model which utilizes advanced GNN techniques assisted by the microbial data to give out the survival prediction for patients.

In our testing, MicrobeSurv demonstrates not only the best performance among other previously published methods but also can provide a good interpretability to show contribution of each microbe.

### Architecture
<img src=imgs/model4.png width=30% />



### Model Trianing and Experiments Repetition

Set up environments:
```bash
conda env create -f microbesurv.yml
conda activate microbesurv
```

1. For benchmarking the performance of our MicrobeSurv model among previous baseline methods, you can execute the following scripts to train and test models, the `-t` parameter specifies how many trials to run 5-fold cross-validation:
```bash
cd src
python run.py -m b3_PFS_cli_g202 -t 100
```

2. For the interpretability of our MicrobeSurv model and visualization of fusion hidden vector or the microbial nodes after graph pooling:
```bash
cd src
python run.py -m b3_PFS_cli_g202 -v
```

### Datasets

Here we supply the training and testing data that already have been preprocessed into pandas dataframe format, which can be found in the `data/processed_data` directory:
- `df_inp_cli_502.pkl`, is the clinical data for 502 patients at total.
- `df_inp_microbes_202.pkl`, is the microbial data including 202 kinds of microbes for each patient.
- `df_inp_graph_202.pkl`, is the interaction network for those microbes.
