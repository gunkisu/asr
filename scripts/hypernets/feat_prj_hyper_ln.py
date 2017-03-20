#!/usr/bin/env python
from __future__ import print_function

import os
import pickle
import argparse
import numpy
import itertools
import theano
import lasagne
from theano import tensor as T
from collections import namedtuple
from libs.lasagne_libs.utils import set_model_param_value

from lasagne.layers import count_params
from lasagne.layers import get_output, get_all_params
from libs.utils import save_network, save_eval_history, best_fer, show_status, symlink_force
from libs.utils import StopWatch, Rsync
from models.gating_hyper_nets import deep_projection_cond_ln_model_fix
from data.wsj.fuel_utils import get_datastream, get_spkid_stream, get_spk_list, spk_to_ids

floatX = theano.config.floatX
eps = numpy.finfo(floatX).eps

input_dim = 123

def add_params(parser):
    parser.add_argument('--batch_size', default=16, help='batch size', type=int)
    parser.add_argument('--num_conds', default=3, help='number of hidden units', type=int)
    parser.add_argument('--num_layers', default=3, help='number of hidden units', type=int)
    parser.add_argument('--num_units', default=512, help='number of hidden units', type=int)
    parser.add_argument('--num_factors', default=64, help='number of factors', type=int)
    parser.add_argument('--learn_rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('--grad_clipping', default=1.0, help='gradient clipping', type=float)
    parser.add_argument('--dropout', default=0.2, help='dropout', type=float)
    parser.add_argument('--data_path', help='data path', default='/data/lisatmp3/speech/tedlium_fbank123_out4174.h5')
    parser.add_argument('--save_path', help='save path', default='./')
    parser.add_argument('--num_epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('--updater', help='sgd or momentum', default='momentum')
    parser.add_argument('--train_disp_freq', help='how ferquently to display progress', default=100, type=int)

    parser.add_argument('--feat_reg', default=0.0, help='feat_reg', type=float)

    parser.add_argument('--train_dataset', help='dataset for training', default='train')
    parser.add_argument('--valid_dataset', help='dataset for validation', default='dev')
    parser.add_argument('--test_dataset', help='dataset for test', default='test')

    parser.add_argument('--reload_model', help='model path to load')
    parser.add_argument('--tmpdir', help='directory name in the /Tmp directory to save data locally',
                        default='/Tmp/taesup/data/speech')

    parser.add_argument('--no-copy', help='do not copy data from NFS to local machine', action='store_true')


def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_params(parser)
    return parser

def get_save_path(args):
    path = args.save_path
    if 'wsj' in args.data_path:
        path += '/wsj_lstmp_uttr_dln'
    else:
        path += '/ted_lstmp_uttr_dln'
    path += '_lr{}'.format(args.learn_rate)
    path += '_gc{}'.format(args.grad_clipping)
    path += '_do{}'.format(args.dropout)
    path += '_nc{}'.format(args.num_conds)
    path += '_nl{}'.format(args.num_layers)
    path += '_nf{}'.format(args.num_factors)
    path += '_nu{}'.format(args.num_units)
    path += '_nb{}'.format(args.batch_size)
    path += '_fr{}'.format(args.feat_reg)

    return path

def build_feat_extractor(input_data,
                         input_mask,
                         output_layer,
                         cond_layer_list):
    get_output(output_layer, deterministic=True)
    utter_feat_list = []
    for cond_layer in cond_layer_list:
        utter_feat_list.append(cond_layer.get_sample_feat())
    utter_feat_list = T.concatenate(utter_feat_list, axis=1)

    return theano.function(inputs=[input_data,
                                   input_mask],
                           outputs=utter_feat_list)

def extarct_feat(extract_fn,
                 data_stream,
                 id_stream):

    data_iterator = data_stream.get_epoch_iterator()
    id_iterator = id_stream.get_epoch_iterator()
    feat_list = []
    id_list = []
    for i, batch_data in enumerate(itertools.izip(data_iterator, id_iterator)):
        data, id = batch_data
        input_data, input_mask, target_data, target_mask = data

        feat_list.append(extract_fn(input_data,
                                    input_mask))
        id_list.append(id[0])
        if i%100==0:
            print('batch process batch {}'.format(i))
    return numpy.concatenate(feat_list, axis=0), numpy.concatenate(id_list, axis=0)

import numpy as Math

def pca(input_data, no_dims=50):
    print("Preprocessing the data using PCA...")
    # normalize
    input_data -= Math.mean(input_data, axis=0, keepdims=True)
    l, M = Math.linalg.eig(Math.dot(input_data.T, input_data))
    Y = Math.dot(input_data, M[:, :no_dims])
    return Y


def Hbeta(D, beta=1.0):
    P = Math.exp(-D.copy() * beta)
    sumP = sum(P)
    H = Math.log(sumP) + beta * Math.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X, tol=1e-5, perplexity=30.0):
	(n, d) = X.shape
	sum_X = Math.sum(Math.square(X), 1)
	D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X)
	P = Math.zeros((n, n))
	beta = Math.ones((n, 1))
	logU = Math.log(perplexity)
	# Loop over all datapoints
	for i in range(n):
		# Print progress
		if i % 500 == 0:
			print("Computing P-values for point ", i, " of ", n, "...")
		betamin = -Math.inf
		betamax = Math.inf
		Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i + 1:n]))]
		(H, thisP) = Hbeta(Di, beta[i])
		Hdiff = H - logU
		tries = 0
		while Math.abs(Hdiff) > tol and tries < 50:
			if Hdiff > 0:
				betamin = beta[i].copy()
				if betamax == Math.inf or betamax == -Math.inf:
					beta[i] = beta[i] * 2
				else:
					beta[i] = (beta[i] + betamax) / 2
			else:
				betamax = beta[i].copy()
				if betamin == Math.inf or betamin == -Math.inf:
					beta[i] = beta[i] / 2
				else:
					beta[i] = (beta[i] + betamin) / 2
			(H, thisP) = Hbeta(Di, beta[i])
			Hdiff = H - logU
			tries = tries + 1
		P[i, Math.concatenate((Math.r_[0:i], Math.r_[i + 1:n]))] = thisP

	print("Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta)))
	return P


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
	# Check inputs
	if isinstance(no_dims, float):
		print("Error: array X should have type float.")
		return -1

	if round(no_dims) != no_dims:
		print("Error: number of dimensions should be an integer.")
		return -1

	# Initialize variables
	X = pca(X, initial_dims).real

	# Init setting
	(n, d) = X.shape
	max_iter = 1000
	initial_momentum = 0.5
	final_momentum = 0.8
	eta = 500
	min_gain = 0.01
	Y = Math.random.randn(n, no_dims)
	dY = Math.zeros((n, no_dims))
	iY = Math.zeros((n, no_dims))
	gains = Math.ones((n, no_dims))

	# Compute P-values
	P = x2p(X, 1e-5, perplexity)
	P = P + Math.transpose(P)
	P = P / Math.sum(P)
	P = P * 4						# early exaggeration
	P = Math.maximum(P, 1e-12)

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = Math.sum(Math.square(Y), 1)
		num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y))
		num[range(n), range(n)] = 0
		Q = num / Math.sum(num)
		Q = Math.maximum(Q, 1e-12)

		# Compute gradient
		PQ = P - Q
		for i in range(n):
			dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
		gains[gains < min_gain] = min_gain
		iY = momentum * iY - eta * (gains * dY)
		Y = Y + iY
		Y = Y - Math.tile(Math.mean(Y, 0), (n, 1))

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = Math.sum(P * Math.log(P / Q))
			print("Iteration ", (iter + 1), ": error is ", C)

		# Stop lying about P-values
		if iter == 100:
			P = P / 4

	# Return solution
	return Y

if __name__ == '__main__':
    ###############
    # load config #
    ###############
    parser = get_arg_parser()
    args = parser.parse_args()
    args.save_path = get_save_path(args)

    if 'wsj' in args.save_path:
        output_dim = 3436
    else:
        output_dim = 4174

    ##############
    # print args #
    ##############
    print(args)

    sw = StopWatch()

    if not args.no_copy:
        print('Loading data streams from {}'.format(args.data_path))
        print('Copying data to local machine...')
        rsync = Rsync(args.tmpdir)
        rsync.sync(args.data_path)
        args.data_path = os.path.join(args.tmpdir, os.path.basename(args.data_path))
        sw.print_elapsed()

    ####################
    # load data stream #
    ####################
    data_stream = get_datastream(path=args.data_path,
                                 which_set=args.test_dataset,
                                 batch_size=args.batch_size,
                                 shuffled=False)
    id_stream = get_spkid_stream(path=args.data_path,
                                 which_set=args.test_dataset,
                                 batch_size=args.batch_size)

    spk_list = get_spk_list(id_stream)
    num_speakers = len(spk_list)
    print('List of speakers: {}'.format(spk_list))

    #################
    # build network #
    #################
    print('Building and compiling network')
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    network_output, cond_layer_list = deep_projection_cond_ln_model_fix(input_var=input_data,
                                                                        mask_var=input_mask,
                                                                        num_inputs=input_dim,
                                                                        num_outputs=output_dim,
                                                                        num_layers=args.num_layers,
                                                                        num_conds=args.num_conds,
                                                                        num_factors=args.num_factors,
                                                                        num_units=args.num_units,
                                                                        grad_clipping=args.grad_clipping,
                                                                        dropout=args.dropout)

    network = network_output
    network_params = get_all_params(network, trainable=True)
    param_count = count_params(network, trainable=True)
    print('Number of parameters of the network: {:.2f}M'.format(float(param_count)/1000000))

    ######################
    # reload model param #
    ######################
    if args.reload_model:
        print('Loading model: {}'.format(args.reload_model))
        with open(args.reload_model, 'rb') as f:
            [pretrain_network_params_val,
             pretrain_update_params_val,
             pretrain_total_batch_cnt] = pickle.load(f)
        set_model_param_value(network_params, pretrain_network_params_val)
    else:
        pretrain_update_params_val = None
        pretrain_total_batch_cnt = 0

    #################
    # build trainer #
    #################
    print('Building extractor')
    sw.reset()
    extract_fn = build_feat_extractor(input_data=input_data,
                                      input_mask=input_mask,
                                      output_layer=network,
                                      cond_layer_list=cond_layer_list)
    sw.print_elapsed()

    ####################
    # start extraction #
    ####################
    print('Starting')

    feat_data, id_data = extarct_feat(extract_fn=extract_fn,
                                      data_stream=data_stream,
                                      id_stream=id_stream)
    id_data = spk_to_ids(spk_list=spk_list,
                         spks=id_data)

    for i in range(3):
        new_feat_data = tsne(X=feat_data[:, i*128:(i+1)*128], no_dims=2, initial_dims=64)
        with open(args.save_path + '_feat{}.pkl'.format(i), 'wb') as f:
            pickle.dump([new_feat_data, id_data], f)
