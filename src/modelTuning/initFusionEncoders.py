# LOAD PACKAGES
import os, sys, time, argparse, subprocess
from itertools import product

from utils import createDirs

# Define current dir
fdir = os.path.dirname(__file__)

# PARSER
parser = argparse.ArgumentParser(description="""Script to initalize the submission of jobs to the queue
                                              for parameter tuning / training out FusionFewShot Feature Extractor.
                                              Parameter lists can be supplied as arguments,
                                              or you may use the provided defaults.""")
parser.add_argument(
    "--ext",
    required=False,
    type=str,
    default='FusionFewShot',
    help='The extension to use as prefix for files and job names\
          to indicate version of model.')
parser.add_argument(
    "--parent",
    required=True,
    type=str,
    help='Location where all the output files should be saved.')
parser.add_argument(
    "--cellPath",
    required=False,
    type=str,
    default=None,
    help='Path to cellLineEncoder')
parser.add_argument(
    "--drugPath",
    required=False,
    type=str,
    default=None,
    help='Path to drugEncoder')
parser.add_argument(
    "--nodeList",
    required=False,
    type=str,
    nargs='+',
    #  default=['128_64_32', '128_32_16', '128_64_16',
            #  '128_64', '128_32', '128_16', 
            #  '64_64_64', '64_64',
            #  '32_32_32', 
    default=['32_32',
            '16_16_16', '16_16', 
            '64_32_16', '64_64_32', '64_64_16',
            '64_32', '64_16', '32_16'],
    help='Number nodes per hidden layer with final item being latent embedding')
parser.add_argument(
    "--activation",
    required=False,
    type=str,
    nargs="+",
    default=['relu', 'sigmoid'],
    help='The hidden layer activation functions to test')
parser.add_argument(
    "--dropout",
    required=False,
    type=float,
    nargs="+",
    default=[0.0, 0.1],
    help='The dropout rates to test in hidden layers')
parser.add_argument(
    "--trainBy",
    required=False,
    type=str,
    nargs="+",
    default=['rna'],
    choices=['rna', 'drug'],
    help="How should support pairs by generated during training.\
            Default is by cell line ('rna'). Opts: ['rna', 'drug']")
parser.add_argument(
    "--learning_rates",
    required=False,
    type=float,
    nargs="+",
    default=[0.01, 0.001],
    help='The learning rate options to test')
parser.add_argument(
    "--decay_rate",
    required=False,
    type=float,
    nargs="+",
    default=[0.99],
    help='The decay rate options to test')
parser.add_argument(
    "--decay_steps",
    required=False,
    type=int,
    nargs="+",
    default=[1024],
    help='The decay step options to test')

#  DEFINE FUNCTIONS
def get_parameter_sets(hypers):
    return list(product(*hypers))

Vector = "list[tuple[str, float, str, float, int, float]]"
def get_flags_jobname(params: Vector, ext: str) -> "tuple[str, str]":
    #  Split parameter set into individual params
    l, do, act, lr, ds, dr, by = params
    
    #  Construct jobname
    mkStr = lambda x: str(x).replace('.','-')
    job = f'NL{l}_DO{mkStr(do)}_AF{act}_LR{mkStr(lr)}_DR{mkStr(dr)}_DS{ds}_BY{by}'
    #  Construct flags
    flags = f"-l {l} -d {do} -a {act} -r {lr} -e {dr} -s {ds} -b {by}"
    flags += f" -o {ext}_{job}"
    return flags, job

def submit_job(ext: str, job: str, flags: str, parent:str):
    queuePath = os.path.join('./', fdir, 'queueFusionEncoder.sh')
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

    nodeList = args.nodeList
    dropout = args.dropout
    activation = args.activation
    learning_rates = args.learning_rates
    decay_steps = args.decay_steps
    decay_rate = args.decay_rate
    trainBy = args.trainBy

    hypers = [nodeList, dropout, activation, 
            learning_rates, decay_steps, decay_rate, trainBy]

    if ~(args.parent.endswith('/')):
        args.parent += '/'

    ext = args.ext
    if args.drugPath == None:
        ext += 'RawDrug'
    else:
        ext += 'EmbedDrug'
    if args.cellPath == None:
        ext += 'RawCell'
    else:
        ext += 'EmbedCell'

    return ext, args.parent, (args.drugPath, args.cellPath), hypers

def run(argv):
    ext, parent, paths, hypers = parse_args(argv)

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
        flags += f' -p {parent} -m {paths[0]} -c {paths[1]}'
        submit_job(ext, job, flags, parent)
        count += 1
    
    print(f'Submitted {count} jobs')


if __name__ == '__main__':
    run(sys.argv[1:])
