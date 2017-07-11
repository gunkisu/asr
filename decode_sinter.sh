#!/usr/bin/env bash
THEANO_FLAGS='device=cpu'
cd ~/song/kaldi/egs/wsj/s5
source ./path.sh

# skip lstm
dir=exp/hyperud/decode_bd_tgpr_dev93_skip_lstm_lr0.01_rlr0.01
model=~/song/asr/skip_lstm_lr0.01_rlr0.01/best_model.ckpt-28056.meta
forward=~/song/asr/bin/forward_skip_lstm.py 
#dir=exp/hyperud/decode_bd_tgpr_dev93_fixed_skip2
#model=~/song/asr/fixed_skip_lstm_wsj_skip2/best_model.ckpt-18698.meta 
#forward=~/song/asr/bin/forward_fixed_skip_lstm.py 

mkdir $dir
python -u $forward --batch-size 1 --metafile $model | latgen-faster-mapped --min-active=200 --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=8.0 --acoustic-scale=0.1 --word-symbol-table=exp/tri4b/graph_bd_tgpr/words.txt exp/hyperud/final.mdl exp/tri4b/graph_bd_tgpr/HCLG.fst ark:- "ark:|gzip -c > $dir/lat.1.gz"
