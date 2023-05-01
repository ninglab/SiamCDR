#LOAD PACKAGES
import argparse, os, sys, timeit
import numpy as np
import pandas as pd

from datetime import datetime
get_time = datetime.now

##  Load from scripts
from utils import createDirs
from contrastiveFusion import FewShotFusion

#### PARSER
parser = argparse.ArgumentParser(usage=__doc__)

## Architecture
parser.add_argument(
    "--drugFewShot",
    required=False,
    type=str,
    default=None,
    help='featureExtractor output dimension')
parser.add_argument(
    "--cellLineFewShot",
    required=False,
    type=str,
    default=None,
    help='featureExtractor output dimension')
parser.add_argument(
    "--trainBy",
    required=False,
    type=str,
    default='rna')
parser.add_argument(
    "--nodeList",
    required=False,
    type=int,
    nargs="+",
    default=32,
    help='Number of nodes per each hidden layer with final layer being latent space embedding')
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
    default=500,
    help='max epochs')
parser.add_argument(
    "--batchSize",
    required=False,
    type=int,
    default=512,
    help='minibatch size')
parser.add_argument(
    "--stepsPerEpoch",
    required=False,
    type=int,
    default=16,
    help='How many steps (batches) per epoch')

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
    default=80,
    help="number of epochs between each decay step of learning rate during training")

### OUT
parser.add_argument(
    "--save",
    required=False,
    type=bool,
    default=False,
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
print("{0}: Executing script: trainFusionEncoder.py".format(get_time()))

# DataPath
drugPath = f'{fdir}/../../data/processed/drug_fingerprints.csv'
cdrPath = f'{fdir}/../../data/processed/drugCellLinePairsData.csv'
rnaPath = f'{fdir}/../../data/processed/RNA_train_cancergenes.csv'

## Functions
def use_parser(argv):
    # Parse input
    args = parser.parse_args(argv)
    inputs = {}

    # Architecture
    inputs['arch'] = {}
    inputs['arch']['drugModelPath'] = args.drugFewShot
    inputs['arch']['cellLineModelPath'] = args.cellLineFewShot
    inputs['arch']['nodeList'] = args.nodeList
    inputs['arch']['dropout'] = args.dropout
    inputs['arch']['activation'] = args.activation

    # Training
    inputs['fit'] = {}
    inputs['fit']['learningRate'] = args.learningRate
    inputs['fit']['decayRate'] = args.decayRate
    inputs['fit']['decaySteps'] = args.decaySteps
    inputs['fit']['epochs'] = args.epochs
    inputs['fit']['batchSize'] = args.batchSize
    inputs['fit']['stepsPerEpoch'] = args.stepsPerEpoch
    
    ## Save
    inputs['out'] = {}
    inputs['out']['save'] = args.save
    inputs['out']['fname'] = args.out
    parent = args.dir
    inputs['out']['paths'] = createDirs(parent).paths
    return inputs


#### MAIN
def run(argv):
    # Parse input fields
    inputs = use_parser(argv)
    
    # Initialize full model
    model = FewShotFusion(**inputs['arch'])
    model_path = os.path.join(inputs['out']['paths']['models'], inputs['out']['fname'])
 
    # Train CDR-smcRBM model
    history = model.fit(rnaPath, drugPath, cdrPath, 
                        saveModel=inputs['out']['save'], 
                        modelPath=model_path,
                        **inputs['fit'])

    sys.stdout.flush()
    print(f'[INFO] {get_time()}: training completed...')

    # Save training and validation loss history
    fit_fname = os.path.join(inputs['out']['paths']['fit'], f"{inputs['out']['fname']}_FitLoss.csv")
    pd.DataFrame({'trainLoss': history['loss']}).to_csv(fit_fname, index=False)

    print(f'[INFO] {get_time()}: training data saved...')
     
    stop = timeit.default_timer()
    print('{1}: Runtime = {0}'.format(stop - start, get_time()))
    sys.stdout.flush()

if __name__ == '__main__':
    run(sys.argv[1:])
    
