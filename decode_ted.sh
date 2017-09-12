#!/usr/bin/env bash
THEANO_FLAGS='device=cpu'
source /u/songinch/.bashrc
source /u/songinch/tensorflow/bin/activate

recipe_dir=~/song/kaldi/egs/tedlium/s5_r2
symbol_table=$recipe_dir/exp/tri3/graph/words.txt
mdl=$recipe_dir/exp/hyperud/final.mdl
hclg=$recipe_dir/exp/tri3/graph/HCLG.fst

source $recipe_dir/path.sh

model=$1
dir=$2
forward=$3
dataset=$4

n_batch=1

mkdir $dir
python -u $forward --n-batch $n_batch --metafile $model --dataset $dataset --no-copy | latgen-faster-mapped --min-active=200 --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=8.0 --acoustic-scale=0.1 --word-symbol-table=$symbol_table $mdl $hclg ark:- "ark:|gzip -c > $dir/lat.1.gz"

