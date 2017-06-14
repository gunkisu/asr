#!/usr/bin/env bash
source /u/songinch/.bashrc
THEANO_FLAGS='device=cpu'
source /u/songinch/tensorflow/bin/activate
cd ~/song/kaldi/egs/wsj/s5
source ./path.sh
python -u ~/song/asr/bin/forward_fixed_skip_lstm.py --batch-size 16 --metafile ~/song/asr/fixed_skip_lstm_wsj_skip0/best_model.ckpt-46745.meta --n-skip 0 | latgen-faster-mapped --beam=13.0 --lattice-beam=8.0 --acoustic-scale=0.1 --word-symbol-table=exp/tri4b/graph_bd_tgpr/words.txt exp/hyperud/final.mdl exp/tri4b/graph_bd_tgpr/HCLG.fst ark:- "ark:|gzip -c > exp/hyperud/decode_bd_tgpr_dev93_skip0/lat.1.gz"
