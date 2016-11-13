#!/bin/bash
#
# Assumes that Kaldi is installed in $KALDI_ROOT and 
# this script is run in $KALDI_ROOT/egs/wsj/s5. 
# You can also create links to the files and directories in $KALDI_ROOT/egs/wsj/s5 instead.
# Required file: path.sh
# Required directories: utils, data-fmllr-tri4b/train_si284_tr90,
# data-fmllr-tri4b/train_si284_cv10, and exp/tri4b_ali_si284.
# It uses kaldi2fuel.py from attention-lvcsr to convert features and targets into a hdf5 file readable by fuel.

. ./path.sh ## Source the tools and utils

datasets=(train_si84, train_si284, test_eval92, test_dev93)

data=data-fmllr-tri4b
dir=exp/hyperud
alidir=exp/tri4b_ali_si284
h5f=exp/hyperud/wsj_fmllr.h5

lvsrdir=/u/songinch/song/attention-lvcsr

stage=0

mkdir -p $dir

. utils/parse_options.sh || exit 1;

# Target generation
if [ $stage -le 0 ]; then
	ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz |" ark,t:- | sort > $dir/all_targets.txt

	# To filter out utterances without targets later
	cut -d ' ' -f 1 $dir/all_targets.txt > $dir/target_uids.txt

	# The number of classes	
  	num_pdf=$(hmm-info $alidir/final.mdl | awk '/pdfs/{print $4}')

	# We need an identity applymap to store targets as an int array in hdf5 
	# through kaldi2fuel.py. There's no other way in kaldi2fuel at the moment.
	for i in $(seq 0 $((num_pdf-1)))
	do
			echo "$i $i" 	
	done > $dir/applymap.txt

	$lvsrdir/bin/kaldi2fuel.py $h5f add_text --applymap $dir/applymap.txt $dir/all_targets.txt targets
fi

# Feature generation
if [ $stage -le 1 ]; then
	for ds in ${datasets[*]}
	do
		cat $data/$ds/feats.scp
	done | sort | uniq > $dir/all_feats.scp

	# Filter out features without targets
	join $dir/target_uids.txt $dir/all_feats.scp > $dir/all_feats_with_targets.scp
	
	compute-global-cmvn-stats.py scp:$dir/all_feats.scp ark:$dir/cmvn-g.stats
	apply-global-cmvn.py --global-stats=ark:$dir/cmvn-g.stats scp:$dir/all_feats_with_targets.scp ark:- | \
		$lvsrdir/bin/kaldi2fuel.py $h5f add ark:- fmllr_feat

fi

# Split generation
if [ $stage -le 2 ]; then
	for ds in ${datasets[*]}
	do
		cat $data/$ds/feats.scp | sort | uniq > $dir/${ds}_feats.scp
		join $dir/target_uids.txt $dir/$ds_feats.scp > $dir/${ds}_feats_with_targets.scp
	done

	for ds in ${datasets[*]}
	do
		echo $ds=$dir/${ds}_feats_with_targets.scp
	done | xargs $lvsrdir/bin/kaldi2fuel.py $h5f split all=$dir/all_feats_with_targets.scp
fi


