# LOAD PACKAGES
import os, sys, time, argparse, subprocess
from itertools import product

from utils import createDirs

# PARSER
parser = argparse.ArgumentParser(description="""Script to initalize the submission of jobs to the queue
                                              for parameter tuning / training our FS-CDR model.
                                              Parameter lists can be supplied as arguments,
                                              or you may use the provided defaults.""")
parser.add_argument(
    "--ext",
    required=False,
    type=str,
    default='FewShotCDR',
    help='The extension to use as prefix for files and job names\
          to indicate version of model.')
parser.add_argument(
    "--parent",
    required=True,
    type=str,
    help='Location where all the output files should be saved.')

# Pretrained encoders
parser.add_argument(
    "--cellEncoder",
    required=False,
    type=str,
    default=None,
    help='Pretrained cell line feature encoder.\
            Default: None (uses raw features)for cell lines')
parser.add_argument(
    "--drugEncoder",
    required=False,
    type=str,
    default=None,
    help='Pretrained drug feature encoder.\
            Default: None (uses raw features)')
parser.add_argument(
    "--fusionEncoder",
    required=False,
    type=str,
    default=None,
    help='Pretrained drug-cell line fusion encoder.\
            Default: None (drug and cell line features simply concatenated)')

## Architecture
parser.add_argument(
    "--nodeList",
    required=False,
    type=str,
    nargs="+",
    default=['64_32_16',  '64_32_8', '64_16_8', '32_16_8', 
            '64_32', '64_16', '64_8', '32_16', '32_8',
            '64', '32',  '64_64', '32_32', '16_16', ''],
    help='Number of nodes in each layer to test')
parser.add_argument(
    "--activation",
    required=False,
    type=str,
    nargs="+",
    default=['relu', 'sigmoid'],
    help='The hidden layer activation functions to test.\
            Default options: [relu, sigmoid]')
parser.add_argument(
    "--dropout",
    required=False,
    type=float,
    nargs="+",
    default=[0.1, 0.3],
    help='The dropout rates to test in hidden layers.\
         Default options: [0.1, 0.3]')
parser.add_argument(
    "--learning_rates",
    required=False,
    type=float,
    nargs="+",
    default=[0.01, 0.001],
    help='The learning rate options to test.\
         Default options: [0.01, 0.001]')
parser.add_argument(
    "--decay_rate",
    required=False,
    type=float,
    nargs="+",
    default=[0.99],
    help='The decay rate options to test. Default: 0.99')
parser.add_argument(
    "--decay_steps",
    required=False,
    type=int,
    nargs="+",
    default=[50, 500],
    help='The decay step options to test.\
            Default options: [50, 500]')

#  DEFINE FUNCTIONS
def get_parameter_sets(hypers):
    return list(product(*hypers))

Vector = "list[tuple[str, int, float, str, float, int, float]]"
def get_flags_jobname(params: Vector, ext: str) -> "tuple[str, str]":
    #  Split parameter set into individual params
    nodeList, do, act, lr, ds, dr = params
    #  Construct jobname
    mkStr = lambda x: str(x).replace('.','-')
    job = f'NL{nodeList}_DO{mkStr(do)}_AF{act}_LR{mkStr(lr)}_DR{mkStr(dr)}_DS{ds}'
    #  Construct flags
    flags = f"-l {nodeList} -d {do} -a {act} -r {lr} -e {dr} -s {ds}"
    flags += f" -o {ext}_{job}"
    return flags, job

def submit_job(ext: str, job: str, flags: str, parent:str):
    cmd = f'sbatch --job-name={ext}_{job}' # submit to sbatch queue
    cmd += f" --output={parent}logs/{ext}_{job}.out" # where to save logfile
    cmd += f' queueFS-CDR.sh {flags}' # which shell script and what flags should be passed to it
    cmd = cmd.split(" ")
    subprocess.run(cmd)

def parse_args(argv):
    args = parser.parse_args(argv)

    nodeList = args.nodeList
    dropout = args.dropout
    activation = args.activation
    learning_rates = args.learning_rates
    decay_steps = args.decay_steps
    decay_rate = args.decay_rate

    hypers = [nodeList, dropout, activation, 
            learning_rates, decay_steps, decay_rate]

    if ~(args.parent.endswith('/')):
        args.parent += '/'

    encoderPaths = [args.drugEncoder, args.cellEncoder, args.fusionEncoder]
    return args.ext, args.parent, hypers, encoderPaths

def run(argv):
    ext, parent, hypers, encoderPaths = parse_args(argv)
    if encoderPaths[2] != None:
        fExt = f'Fused'
    else:
        fExt = f'Unfused'
    if encoderPaths[0] == None:
        dExt = 'RawDrug'
    else:
        dExt = 'EmbedDrug'
    if encoderPaths[1] == None:
        rExt = 'RawCell'
    else:
        rExt = 'EmbedCell'
    ext = f"{fExt}-{ext}{dExt}{rExt}"

    constantFlags = f" -g {encoderPaths[0]} -c {encoderPaths[1]} -f {encoderPaths[2]}"
    constantFlags += f" -p {parent}"

    nArch = 1
    for h in hypers:
        nArch *= len(h)

    print(f"\nSubmitting {nArch} jobs to conduct parameter tuning...")
    print("Will attempt to submit all jobs in 5s. Abort now if undesired...")
    time.sleep(5)
    print("Proceeding...\n")

    dirs = createDirs(parent)
    logdir = os.path.join(dirs.parent, 'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    count = 0
    param_sets = get_parameter_sets(hypers)
    for param_set in param_sets:
        flags, job = get_flags_jobname(param_set, ext)
        flags += constantFlags
        submit_job(ext, job, flags, parent)
        count += 1
    
    print(f'Submitted {count} jobs')


if __name__ == '__main__':
    run(sys.argv[1:])
