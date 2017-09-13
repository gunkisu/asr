#!/usr/bin/env bash
THEANO_FLAGS='device=cpu'
source /u/songinch/.bashrc
source /u/songinch/tensorflow/bin/activate

recipe_dir=~/song/kaldi/egs/wsj/s5
symbol_table=$recipe_dir/exp/tri4b/graph_bd_tgpr/words.txt
mdl=$recipe_dir/exp/hyperud/final.mdl
hclg=$recipe_dir/exp/tri4b/graph_bd_tgpr/HCLG.fst

datapath=~/song/data/speech/wsj_fbank123.h5

cd $recipe_dir
source ./path.sh

model=$1
dir=$2
forward=$3
dataset=$4

n_batch=1

mkdir $dir
python -u $forward --n-batch $n_batch --metafile $model --data-path $datapath --dataset $dataset --no-copy | latgen-faster-mapped --min-active=200 --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=8.0 --acoustic-scale=0.1 --word-symbol-table=$symbol_table $mdl $hclg ark:- "ark:|gzip -c > $dir/lat.1.gz"

