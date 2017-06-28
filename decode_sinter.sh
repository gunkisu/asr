#!/usr/bin/env bash
THEANO_FLAGS='device=cpu'
cd ~/song/kaldi/egs/wsj/s5
source ./path.sh
mkdir exp/hyperud/decode_bd_tgpr_dev93_skip_lstm
python -u ~/song/asr/bin/forward_skip_lstm.py --batch-size 4 --metafile ~/song/asr/skip_lstm/best_model.ckpt-14040.meta | latgen-faster-mapped --min-active=200 --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=8.0 --acoustic-scale=0.1 --word-symbol-table=exp/tri4b/graph_bd_tgpr/words.txt exp/hyperud/final.mdl exp/tri4b/graph_bd_tgpr/HCLG.fst ark:- "ark:|gzip -c > exp/hyperud/decode_bd_tgpr_dev93_skip_lstm/lat.1.gz"
