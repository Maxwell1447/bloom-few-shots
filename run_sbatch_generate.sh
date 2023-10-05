#!/bin/bash

sbatch --job-name=gen_bloom --output=../logs/generate%j.out --error=../logs/generate.%j.log --export=ALL, sbatch_generate.sh