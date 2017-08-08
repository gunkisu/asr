#!/usr/bin/env bash
source /u/songinch/.bashrc
THEANO_FLAGS='device=cpu'
source /u/songinch/tensorflow/bin/activate
cd ~/song/kaldi/egs/wsj/s5
source ./path.sh

# skiprnn trained by reinforce
dir=exp/hyperud/decode_bd_tgpr_dev93_skiprnn_ri_a6_y_e32
model=~/song/asr/skiprnn_ri_a6_y_e32/best_model.ckpt-7014.meta
forward=~/song/asr/bin/forward_skiprnn_ri.py

# skiprnn with subsampling
dir=exp/hyperud/decode_bd_tgpr_dev93_skiprnn_subsample_s0
model=~/song/asr/skiprnn_subsample_s0_lr0.01/best_model.ckpt-11690.meta
forward=~/song/asr/bin/forward_skiprnn_subsample.py

# skip lstm supervised
#dir=exp/hyperud/decode_bd_tgpr_dev93_skip_lstm_supervised_a7
#model=~/song/asr/skip_lstm_supervised_a7/best_model.ckpt-32732.meta
#forward=~/song/asr/bin/forward_skip_lstm_supervised.py 

n_batch=1

mkdir $dir
python -u $forward --n-batch $n_batch --metafile $model --no-copy | latgen-faster-mapped --min-active=200 --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=8.0 --acoustic-scale=0.1 --word-symbol-table=exp/tri4b/graph_bd_tgpr/words.txt exp/hyperud/final.mdl exp/tri4b/graph_bd_tgpr/HCLG.fst ark:- "ark:|gzip -c > $dir/lat.1.gz"

