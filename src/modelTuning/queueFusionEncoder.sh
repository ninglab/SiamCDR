#!/bin/bash

#SBATCH --account=PCON0041

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=ALL
#SBATCH	--mail-user=patrick.skillman-lawrence@osumc.edu

##### Main
batch=32
steps=1024
epoch=250

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

echo $out':'
nodeList=${nodeList//'_'/' '}
python ../models/trainFusionEncoder.py --nodeList ${nodeList}\
                --cellLineFewShot ${cellEncoder} --drugFewShot ${drugEncoder}\
                --dropout ${dropout} --activation ${act} --learningRate ${lr}\
                --decayRate ${dr} --decaySteps ${ds} --epochs ${epoch}\
                --trainBy ${by} --batchSize ${batch} --stepsPerEpoch ${steps}\
                --dir ${parent} --save True --out ${out}
printf '\n\n\n'
