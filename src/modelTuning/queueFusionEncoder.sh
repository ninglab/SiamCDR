#!/bin/bash

#SBATCH --account=PCON0041

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

##### Main
## Get script location to define training file path
if [[ -n $SLURM_JOB_ID ]] ; then
    SCRIPT_DIR=$(realpath $(dirname $(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | cut -d" " -f1)))
else
    SCRIPT_DIR=$(dirname $(realpath $0))
fi
MODEL_DIR=${SCRIPT_DIR/Tuning/s}

## define variables
batch=32
steps=1024
epoch=250

## register flags / arguments
while getopts l:b:d:a:p:r:e:c:m:s:o: flag
do
    case "${flag}" in
        p) parent=${OPTARG};;
        l) nodeList=${OPTARG};;
        d) dropout=${OPTARG};;
        b) by=${OPTARG};;
        a) act=${OPTARG};;
        r) lr=${OPTARG};;
        e) dr=${OPTARG};;
        s) ds=${OPTARG};;
        c) cellEncoder=${OPTARG};;
        m) drugEncoder=${OPTARG};;
        o) out=${OPTARG};;
    esac
done

## submit job
echo $out':'
nodeList=${nodeList//'_'/' '}
python ${MODEL_DIR}/trainFusionEncoder.py --nodeList ${nodeList}\
                --cellLineFewShot ${cellEncoder} --drugFewShot ${drugEncoder}\
                --dropout ${dropout} --activation ${act} --learningRate ${lr}\
                --decayRate ${dr} --decaySteps ${ds} --epochs ${epoch}\
                --trainBy ${by} --batchSize ${batch} --stepsPerEpoch ${steps}\
                --dir ${parent} --save True --out ${out}
printf '\n\n'
