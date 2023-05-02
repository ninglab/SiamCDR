#LOAD PACKAGES
import argparse, os, sys, timeit
import pandas as pd

from datetime import datetime
get_time = datetime.now

##  Load from scripts
from utils import createDirs, Predictions
from tf_utils import DataLoader
from fsCDR import fsCDR

#### PARSER
parser = argparse.ArgumentParser(usage=__doc__)

### Model Architecture
## pretrained models
parser.add_argument(
    "--cellEncoder",
    required=False,
    type=str,
    default=None,
    help='Pretrained feature encoder for cell lines')
parser.add_argument(
    "--drugEncoder",
    required=False,
    type=str,
    default=None,
    help='Pretrained feature encoder for drugs')
parser.add_argument(
    "--fusionEncoder",
    required=False,
    type=str,
    default=None,
    help='Pretrained feature encoder for cell line drug pair fusion')

## Architecture
parser.add_argument(
    "--nodeList",
    required=False,
    type=int,
    nargs='+',
    default=[],
    help='Number of nodes in each hidden layer')
parser.add_argument(
    "--dropout",
    required=False,
    type=float,
    default=0.1,
    help='dropout rate')
parser.add_argument(
    "--activation",
    required=False,
    type=str,
    default='relu',
    help="activation function to use for the model's hidden layers")

### Model Training
## early stopping
parser.add_argument(
    "--epochs",
    required=False,
    type=int,
    default=1000,
    help='max epochs')
parser.add_argument(
    "--batchSize",
    required=False,
    type=int,
    default=512,
    help='minibatch size')
parser.add_argument(
    "--patience",
    required=False,
    type=int,
    default=15,
    help='Patience for early stopping')
parser.add_argument(
    "--minDelta",
    required=False,
    type=float,
    default=0.0001,
    help='Minimum change in validation AUC to reset early stopping patience counter')

## learning rate schedule
parser.add_argument(
    "--learningRate",
    required=False,
    type=float,
    default=0.001,
    help='starting learning rate for training')
parser.add_argument(
    "--decayRate",
    required=False,
    type=float,
    default=0.99,
    help="rate of decay for training's exponential decay learning rate scheduler")
parser.add_argument(
    "--decaySteps",
    required=False,
    type=int,
    default=50,
    help="number of epochs between each decay step of learning rate during training")

### OUT
parser.add_argument(
    "--save",
    required=False,
    type=bool,
    default=True,
    help='Should weights be saved after fit to data?')
parser.add_argument(
    "--dir",
    type=str,
    default='full_model',
    help='parent directory for saving output from training and testing')
parser.add_argument(
    "--out",
    required=True,
    type=str,
    help='file name for saving results and model')


#### DEFINE
### Constants
fdir = os.path.dirname(__file__)

start = timeit.default_timer()
print("{0}: Executing script: trainFS-CDR.py".format(get_time()))

rna_paths = {'train': f'{fdir}/../../data/processed/RNA_train_cancergenes.csv',
             'val': f'{fdir}/../../data/processed/RNA_val_cancergenes.csv',
             'test': f'{fdir}/../../data/processed/RNA_test_cancergenes.csv',
             'newcancer': f'{fdir}/../../data/processed/RNA_newcancer_cancergenes.csv'}

drug_path = f'{fdir}/../../data/processed/drug_fingerprints.csv'
cdr_path = f'{fdir}/../../data/processed/drugCellLinePairsData.csv'
info_path = f'{fdir}/../../data/processed/cellLineInfo.csv'

## Functions
def use_parser(argv):
    # Parse input
    args = parser.parse_args(argv)
    inputs = {}

    # Architecture
    inputs['arch'] = {}
    inputs['arch']['cellLineModelPath'] = args.cellEncoder
    inputs['arch']['drugModelPath'] = args.drugEncoder
    inputs['arch']['nodeList'] = args.nodeList
    print(f"[INFO] Node List: {args.nodeList}...")
    inputs['arch']['dropout'] = args.dropout
    inputs['arch']['activation'] = args.activation
    inputs['arch']['fusionModelPath'] = args.fusionEncoder

    # Training
    inputs['fit'] = {}
    inputs['fit']['learningRate'] = args.learningRate
    inputs['fit']['decayRate'] = args.decayRate
    inputs['fit']['decaySteps'] = args.decaySteps
    inputs['fit']['epochs'] = args.epochs
    inputs['fit']['batchSize'] = args.batchSize
    inputs['fit']['patience'] = args.patience
    inputs['fit']['minDelta'] = args.minDelta

    ## Save
    inputs['out'] = {}
    inputs['out']['save'] = args.save
    inputs['out']['fname'] = args.out

    parent = args.dir
    inputs['out']['paths'] = createDirs(parent).paths

    metrics = ['precision', 'recall', 'f1']

    return inputs, metrics

#### MAIN
def run(argv):
    # Parse input fields
    inputs, metrics = use_parser(argv)

    # Load data
    data = {}
    cdrs = {}
    loader = DataLoader(drug_path, cdr_path, info_path, drugsProcessed=True)
    for split, path in rna_paths.items():
        data[split] = {}
        data[split]['rna'], data[split]['drug'], data[split]['GT'], cdrs[split] = loader.get_split(path)
    
    # Initialize full model
    model = fsCDR(**inputs['arch'])
 
    # Train CDR-smcRBM model
    model_path = os.path.join(inputs['out']['paths']['models'], inputs['out']['fname'])
    history = model.fit(train=(data['train']['drug'], data['train']['rna'], data['train']['GT']),
                        val=(data['val']['drug'], data['val']['rna'], data['val']['GT']),
                        modelPath=model_path, saveModel=inputs['out']['save'], **inputs['fit'])

    sys.stdout.flush()
    print(f'{get_time()}: Training completed.')

    # Save training and validation loss history
    fit_fname = os.path.join(inputs['out']['paths']['fit'], f"{inputs['out']['fname']}_FitLoss.csv")
    pd.DataFrame({'train_loss': history['loss'],
                 'val_loss': history['val_loss']}).to_csv(fit_fname, index=False)
    print(f'{get_time()}: Training data saved')
     
    print(f'{get_time()}: Obtaining predictions and performance metrics')
    evaluator = Predictions(model)
    for split, cdr in cdrs.items():
        input_data = [data[split]['drug'], data[split]['rna']]
        cancer_type = cdr.DepMap_ID.apply(loader.cancer_lookup).values
        res_out = os.path.join(inputs['out']['paths'][split+'_res'], inputs['out']['fname'])
        if split in ['train', 'val']:
            evaluator.evalPerformance(input_data, cdr, cancer_type, metrics, res_out, save_preds=False)
        else:
            pred_out = os.path.join(inputs['out']['paths'][split+'_pred'], inputs['out']['fname'])
            evaluator.evalPerformance(input_data, cdr, cancer_type, metrics, res_out, pred_out, save_preds=True)

    # save average performance of k fold cross validation
    stop = timeit.default_timer()
    print('{1}: Runtime = {0}'.format(stop - start, get_time()))
    sys.stdout.flush()

if __name__ == '__main__':
    run(sys.argv[1:])
    
