#!/usr/bin/env bash
source /u/songinch/.bashrc
THEANO_FLAGS='device=cpu'
source /u/songinch/tensorflow/bin/activate
cd ~/song/kaldi/egs/wsj/s5
source ./path.sh


dir=$1
model=$2
forward=$3

# skiprnn trained by reinforce
#dir=exp/hyperud/decode_bd_tgpr_dev93_skiprnn_ri_a6_fa20
#model=~/song/asr/skiprnn_ri_a6_fa20/best_model.ckpt-21042.meta
#forward=~/song/asr/bin/forward_skiprnn_ri.py

# skiprnn with subsampling
#dir=exp/hyperud/decode_bd_tgpr_dev93_skiprnn_subsample_s4
#model=~/song/asr/skiprnn_subsample_s4/best_model.ckpt-28056.meta
#forward=~/song/asr/bin/forward_skiprnn_subsample.py

# skiprnn trained with supervised training
#dir=exp/hyperud/decode_bd_tgpr_dev93_skiprnn_sv_a8
#model=~/song/asr/skiprnn_sv_a8/best_model.ckpt-30394.meta
#forward=~/song/asr/bin/forward_skiprnn_sv.py

n_batch=1

mkdir $dir
python -u $forward --n-batch $n_batch --metafile $model --no-copy | latgen-faster-mapped --min-active=200 --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=8.0 --acoustic-scale=0.1 --word-symbol-table=exp/tri4b/graph_bd_tgpr/words.txt exp/hyperud/final.mdl exp/tri4b/graph_bd_tgpr/HCLG.fst ark:- "ark:|gzip -c > $dir/lat.1.gz"

