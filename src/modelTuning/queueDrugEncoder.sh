#!/bin/bash

#SBATCH --account=PCON0041

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=16gb

#SBATCH --mail-type=ALL
#SBATCH	--mail-user=patrick.skillman-lawrence@osumc.edu

##### Main
batch=512
epoch=1000
patience=10
min_delta=0.0001

while getopts l:n:d:a:p:r:e:s:c:o: flag
do
    case "${flag}" in
        p) parent=${OPTARG};;
        l) layers=${OPTARG};;
        n) n_hidden=${OPTARG};;
        d) dropout=${OPTARG};;
        a) act=${OPTARG};;
        c) cont=${OPTARG};;
        r) lr=${OPTARG};;
        e) dr=${OPTARG};;
        s) ds=${OPTARG};;
        o) out=${OPTARG};;
    esac
done

echo $out':'
python ../models/trainDrugEncoder.py --embeddingDim ${n_hidden} --nHiddenLayers ${layers}\
                --dropout ${dropout} --activation ${act} --learningRate ${lr}\
                --decayRate ${dr} --decaySteps ${ds} --epochs ${epoch} --batchSize ${batch}\
                --patience ${patience} --minDelta ${min_delta} --continuous ${cont}\
                --dir ${parent} --save True --out ${out}
printf '\n\n\n'
