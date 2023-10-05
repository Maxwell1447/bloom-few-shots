#!/bin/bash

##### generate
##SBATCH --partition=gpu_p2
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1                  # nombre de taches a reserver (=nombre de GPU ici)
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=3            # nombre de coeurs CPU par tache (un quart du noeud ici)
##SBATCH --time=10:00:00
##SBATCH --qos=qos_gpu-t3            # QoS determines the time/node limits, and priority of your job

##### debug
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                  # nombre de taches a reserver (=nombre de GPU ici)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3          # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH --time=00:05:00
#SBATCH --qos=qos_gpu-dev             # QoS determines the time/node limits, and priority of your job

# activate conda env
source /gpfswork/rech/usb/ufn16wp/miniconda3/bin/activate bloom

# MODEL_PATH=/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-560m
MODEL_PATH=/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-3b

GEN_OPTION=
# maxk=-1
# maxk=-2
maxk=-3

# RETRIEVAL_OPTION=normal-0.3
RETRIEVAL_OPTION=no-filter-fuzzy-0$maxk

# PATH to prompt data file
CTX_DIR=/gpfswork/rech/usb/ufn16wp/NLP4NLP/DATA/multi-domain/data-llm-few-shot/en-fr-$RETRIEVAL_OPTION.ctx
# PATH to directory of generated files
GEN_DIR=/gpfswork/rech/usb/ufn16wp/NLP4NLP/DATA/multi-domain/generated/en-fr-bloom/en-fr-$RETRIEVAL_OPTION

mkdir -p $GEN_DIR

# 10hrs for 1 test file

for name in ECB # EMEA Europarl GNOME JRC-Acquis KDE4 News-Commentary PHP TED2013 Ubuntu Wikipedia
do
    for m in \<0.6 # \>0.6
    do
        # Prompt sentences file
        file=$CTX_DIR/${name}_0.4+bin$maxk.test$m.en-fr.ctx
        # Output sentences file
        out=$GEN_DIR/${name}$GEN_OPTION.test$m.hyp.fr

        if [[ -f $GEN_DIR/${name}$GEN_OPTION.test$m.ref.fr ]]; then
            rm -f $GEN_DIR/${name}$GEN_OPTION.test$m.ref.fr
        fi;
        
        # symbolic link reference file to generation directory
        ln -s $CTX_DIR/${name}_0.4+bin.test$m.en-fr.fr $GEN_DIR/${name}$GEN_OPTION.test$m.ref.fr
        
        python generate.py \
        --model-path $MODEL_PATH \
        --file $file \
        --out $out \
        --tqdm \
        --max-new-tokens 250 \
        # --debug

    done;
done;