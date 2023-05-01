#LOAD PACKAGES
import argparse, os, sys, timeit
import pandas as pd

from datetime import datetime
get_time = datetime.now

##  Load from scripts
from utils import createDirs
from siamese import SiameseNeuralNet

#### PARSER
parser = argparse.ArgumentParser(usage=__doc__)

## Architecture
parser.add_argument(
    "--embeddingDim",
    required=False,
    type=int,
    default=32,
    help='featureExtractor output dimension')
parser.add_argument(
    "--nHiddenLayers",
    required=False,
    type=int,
    default=1,
    help='Number of hidden layers in featureExtractor')
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
    default=200,
    help='max epochs')
parser.add_argument(
    "--batchSize",
    required=False,
    type=int,
    default=64,
    help='minibatch size')
parser.add_argument(
    "--patience",
    required=False,
    type=int,
    default=10,
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
    default=1000,
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
print("{0}: Executing script: trainCellLineEncoder.py".format(get_time()))

# DataPath
rnaPath = f'{fdir}/../../data/processed/cellLineEncoderPretrainData.csv'
infoPath = f'{fdir}/../../data/processed/cellLineInfo.csv'

## Functions
def use_parser(argv):
    # Parse input
    args = parser.parse_args(argv)
    inputs = {}

    # Architecture
    inputs['arch'] = {}
    inputs['arch']['nHiddenLayers'] = args.nHiddenLayers
    inputs['arch']['embeddingDim'] = args.embeddingDim
    inputs['arch']['dropout'] = args.dropout
    inputs['arch']['activation'] = args.activation

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
    return inputs

#### MAIN
def run(argv):
    # Parse input fields
    inputs = use_parser(argv)
    
    # Initialize full model
    model = SiameseNeuralNet(inputShape=463, **inputs['arch'])
 
    # Train cell line encoder model
    model_path = os.path.join(inputs['out']['paths']['models'], inputs['out']['fname'])
    history = model.fit(rnaPath, 
                        saveModel=inputs['out']['save'], 
                        modelPath=model_path,
                        **inputs['fit'])

    sys.stdout.flush()
    print(f'[INFO] {get_time()}: training completed...')

    # Save training and validation loss history
    fit_fname = os.path.join(inputs['out']['paths']['fit'], f"{inputs['out']['fname']}_FitLoss.csv")
    pd.DataFrame({'trainLoss': history['loss'],
                  'valLoss': history['val_loss'],
                  'trainAcc': history['accuracy'],
                  'valAcc': history['val_accuracy']}).to_csv(fit_fname, index=False)
    print(f'{get_time()}: Training data saved')
     
    stop = timeit.default_timer()
    print('{1}: Runtime = {0}'.format(stop - start, get_time()))
    sys.stdout.flush()

if __name__ == '__main__':
    run(sys.argv[1:])
    
