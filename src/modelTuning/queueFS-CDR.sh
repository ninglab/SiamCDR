#!/bin/bash

#SBATCH --account=PCON0041

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=ALL
#SBATCH	--mail-user=patrick.skillman-lawrence@osumc.edu

##### Main
batch=256
epoch=1000
patience=10
min_delta=0.0001

while getopts l:d:a:p:r:e:s:c:g:b:f:o: flag
do
    case "${flag}" in
        p) parent=${OPTARG};;
        c) cellEncoder=${OPTARG};;
        g) drugEncoder=${OPTARG};;
        f) fusionEncoder=${OPTARG};;
        b) fuse=${OPTARG};;
        l) nodeList=${OPTARG};;
        d) dropout=${OPTARG};;
        a) act=${OPTARG};;
        r) lr=${OPTARG};;
        e) dr=${OPTARG};;
        s) ds=${OPTARG};;
        o) out=${OPTARG};;
    esac
done

echo $out':'
if [[ $nodeList == '' ]]; then
    python ../models/trainFS-CDR.py --cellEncoder ${cellEncoder} --drugEncoder ${drugEncoder}\
                --fusion ${fuse} --fusionEncoder ${fusionEncoder}\
                --dropout ${dropout} --activation ${act} --learningRate ${lr}\
                --decayRate ${dr} --decaySteps ${ds} --epochs ${epoch} --batchSize ${batch}\
                --patience ${patience} --minDelta ${min_delta}\
                --dir ${parent} --save True --out ${out}
else
    nodeList=${nodeList//'_'/' '}
    python ../models/trainFS-CDR.py --cellEncoder ${cellEncoder} --drugEncoder ${drugEncoder}\
                --fusion ${fuse} --fusionEncoder ${fusionEncoder} --nodeList ${nodeList} \
                --dropout ${dropout} --activation ${act} --learningRate ${lr}\
                --decayRate ${dr} --decaySteps ${ds} --epochs ${epoch} --batchSize ${batch}\
                --patience ${patience} --minDelta ${min_delta}\
                --dir ${parent} --save True --out ${out}
fi
printf '\n\n\n'
