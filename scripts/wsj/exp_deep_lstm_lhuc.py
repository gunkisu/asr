#!/usr/bin/env python
from __future__ import print_function

import numpy, theano, lasagne, pickle, os
from theano import tensor as T
from collections import OrderedDict, namedtuple

from libs.deep_lstm_utils import *
import libs.deep_lstm_utils as deep_lstm_utils
from libs.lasagne_libs.updates import momentum

import libs.utils as utils
import models.deep_bidir_lstm as models
import data.wsj.fuel_utils as fuel_utils
import libs.param_utils as param_utils

import data.transformers as trans
import itertools
from fuel.transformers import Padding

if __name__ == '__main__':
    parser = deep_lstm_utils.get_arg_parser()
    deep_lstm_utils.add_lhuc_params(parser)
    args = parser.parse_args()

    args.save_path = get_save_path(args)    
    args.save_path = '{}_lhuc'.format(args.save_path)

    if args.use_ivectors:
        args.input_dim = args.input_dim + args.ivector_dim

    print(args)

    print('Load data stream {} from {}'.format(args.valid_dataset, 
        args.data_path))
    if args.norm_path: 
        print('Use normalization data from {}'.format(args.norm_path))
    
    valid_ds = fuel_utils.get_datastream(path=args.data_path,
                                  which_set=args.valid_dataset,
                                  batch_size=args.batch_size, 
                                  norm_path=args.norm_path,
                                  use_ivectors=args.use_ivectors,
                                  truncate_ivectors=args.truncate_ivectors,
                                  ivector_dim=args.ivector_dim)
    valid_spkid_ds = fuel_utils.get_spkid_stream(path=args.data_path, 
        which_set=args.valid_dataset, batch_size=args.batch_size)

    
    spk_list = fuel_utils.get_spk_list(valid_spkid_ds)
    num_speakers = len(spk_list) 
    print('List of speakers: {}'.format(spk_list))

    print('Build and compile network')
    input_data = T.ftensor3('input_data')
    input_mask = T.fmatrix('input_mask')
    target_data = T.imatrix('target_data')
    target_mask = T.fmatrix('target_mask')
    speaker_data = T.ivector('speaker_data')

    
    network = models.deep_bidir_lstm_lhuc(input_var=input_data,
                                    mask_var=input_mask,
                                    input_dim=args.input_dim,
                                    num_units_list=[args.num_nodes]*args.num_layers,
                                    output_dim=args.output_dim,
                                    speaker_var=speaker_data, num_speakers=num_speakers)

    print('Loading the speaker independent model: {}'.format(args.simodel))

    # Get params except those of LHUC layers
    si_net_params = get_all_params(network, trainable=True, speaker_dependent=False)

    with open(args.simodel, 'rb') as f:
        pre_si_net_params, pre_si_update_params, pre_epoch_no_ = pickle.load(f)
        param_utils.set_model_param_value(si_net_params, pre_si_net_params)
    
    network_params = get_all_params(network, trainable=True)

    print('Build trainer')
    sw = utils.StopWatch()
    training_fn, trainer_params = trainer_lhuc(
              input_data=input_data,
              input_mask=input_mask,
              target_data=target_data,
              target_mask=target_mask,
              speaker_data=speaker_data,
              network=network,
              updater=eval(args.updater),
              learning_rate=args.learn_rate,
              load_updater_params=None)
    sw.print_elapsed()

    print('Build predictor')
    sw.reset()
    predict_fn = predictor_lhuc(input_data=input_data,
                                       input_mask=input_mask,
                                       target_data=target_data,
                                       target_mask=target_mask,
                                       speaker_data=speaker_data,
                                       network=network)

    sw.print_elapsed()

    print('Start training')
    EvalRecord = namedtuple('EvalRecord', ['ce_frame', 'fer'])

    eval_history =[EvalRecord(10.0, 1.0)]
    early_stop_flag = False
    early_stop_cnt = 0

    # e_idx starts from 1
    for e_idx in range(1, args.num_epochs+1):
        epoch_sw = utils.StopWatch()
        print('--')
        print('Epoch {} starts'.format(e_idx))
        print('--')
        
        train_ce_frame_sum = 0.0
        for b_idx, lhuc_data in enumerate(itertools.izip(valid_ds.get_epoch_iterator(), 
                valid_spkid_ds.get_epoch_iterator()), start=1):
            data, spk_data = lhuc_data
            input_data, input_mask, target_data, target_mask = data
            train_output = training_fn(input_data,
                                       input_mask,
                                       target_data,
                                       target_mask, fuel_utils.spk_to_ids(spk_list, spk_data[0]))
            ce_frame = train_output[0]
            network_grads_norm = train_output[1]

            if b_idx%args.train_disp_freq == 0: 
                show_status(args.save_path, ce_frame, network_grads_norm, b_idx, args.batch_size, e_idx)
            train_ce_frame_sum += ce_frame
            
        print('End of Epoch {}'.format(e_idx))
        epoch_sw.print_elapsed()
        
        print('Evaluating the network on the validation dataset')
        eval_sw = utils.StopWatch()
        #train_ce_frame, train_fer = eval_net(predict_fn, train_ds)
        valid_ce_frame, valid_fer = eval_net_lhuc(predict_fn, valid_ds, valid_spkid_ds, spk_list)
        eval_sw.print_elapsed()

        if valid_fer>eval_history[-1].fer:
            early_stop_cnt += 1.
        else:
            early_stop_cnt = 0.
            save_network(network_params, trainer_params, e_idx, args.save_path+'_best_model.pkl') 

        print('Train CE: {}'.format(train_ce_frame_sum / b_idx))
        print('Valid CE: {}, FER: {}'.format(valid_ce_frame, valid_fer))

        print('Saving the network and evaluation history')
        eval_history.append(EvalRecord(valid_ce_frame, valid_fer))
        numpy.savez(args.save_path + '_eval_history',
                    eval_history=eval_history)

        save_network(network_params, trainer_params, e_idx, args.save_path + '_last_model.pkl')
        
        if early_stop_cnt>10:
            print('Training early stopped')
            break



