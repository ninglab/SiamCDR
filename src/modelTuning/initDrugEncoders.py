# LOAD PACKAGES
import os, sys, time, argparse, subprocess
from itertools import product

from utils import createDirs

# Define current dir
fdir = os.path.dirname(__file__)

# PARSER
parser = argparse.ArgumentParser(description="""Script to initalize the submission of jobs to the queue
                                              for parameter tuning / training out Drug FewShot Feature Extractor.
                                              Parameter lists can be supplied as arguments,
                                              or you may use the provided defaults.""")
parser.add_argument(
    "--ext",
    required=False,
    type=str,
    default='DrugEncoder',
    help='The extension to use as prefix for files and job names\
          to indicate version of model.')
parser.add_argument(
    "--parent",
    required=True,
    type=str,
    help='Location where all the output files should be saved.')
parser.add_argument(
    "--layers",
    required=False,
    type=int,
    nargs='+',
    default=[1, 2],
    help='Number of hidden layers to test\
            Default options: [1, 2]')
parser.add_argument(
    "--n_hidden",
    required=False,
    type=int,
    nargs="+",
    default=[64, 32, 16],
    help='Number of hidden nodes in each layer to test.\
            Output embedding dimensions is equal to this.\
            Default options: [64, 32, 16]')
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
    default=[0.01, 0.001, 0.0001],
    help='The learning rate options to test.\
         Default options: [0.01, 0.001, 0.0001]')
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
    default=[1024],
    help='The decay step options to test. Default: 1024')


#  DEFINE FUNCTIONS
def get_parameter_sets(hypers):
    return list(product(*hypers))

Vector = "list[tuple[int, int, float, str, float, int, float]]"
def get_flags_jobname(params: Vector, ext: str) -> "tuple[str, str]":
    #  Split parameter set into individual params
    l, n, do, act, lr, ds, dr = params
    
    #  Construct jobname
    mkStr = lambda x: str(x).replace('.','-')
    job = f'Layers{l}_Hidden{n}_DO{mkStr(do)}_AF{act}_LR{mkStr(lr)}_DR{mkStr(dr)}_DS{ds}'
    #  Construct flags
    flags = f"-l {l} -n {n} -d {do} -a {act} -r {lr} -e {dr} -s {ds}"
    flags += f" -o {ext}_{job}"
    return flags, job

def submit_job(ext: str, job: str, flags: str, parent:str):
    queuePath = os.path.join('./', fdir, 'queueDrugEncoder.sh')
    #  cmd = 'bash'  # Use if want to test how command is being submitted
    cmd = f'sbatch --job-name={ext}_{job}' # submit to sbatch queue
    cmd += f" --output={parent}logs/{ext}_{job}.out" # where to save logfile
    cmd += f' {queuePath} {flags}' # which shell script and what flags should be passed to it
    #  print(cmd)
    cmd = cmd.split(" ")
    subprocess.run(cmd)
    #  sys.exit()

def parse_args(argv):
    args = parser.parse_args(argv)

    layers = args.layers
    n_hidden = args.n_hidden
    dropout = args.dropout
    activation = args.activation
    learning_rates = args.learning_rates
    decay_steps = args.decay_steps
    decay_rate = args.decay_rate

    hypers = [layers, n_hidden, dropout, activation, 
            learning_rates, decay_steps, decay_rate]

    if ~(args.parent.endswith('/')):
        args.parent += '/'

    return args.ext, args.parent, args.continuous, hypers

def run(argv):
    ext, parent, cont, hypers = parse_args(argv)

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
        flags += f' -c {cont} -p {parent}'
        submit_job(ext, job, flags, parent)
        count += 1
    
    print(f'Submitted {count} jobs')


if __name__ == '__main__':
    run(sys.argv[1:])
