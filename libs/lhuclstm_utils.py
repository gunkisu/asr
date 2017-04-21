import argparse
import os

def add_params(parser):
    parser.add_argument('--batch-size', default=2, help='batch size', type=int)
    parser.add_argument('--num-units', default=10, help='number of hidden units', type=int)
    parser.add_argument('--num-layers', default=1, help='number of layers', type=int)
    parser.add_argument('--learn-rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('--grad-clipping', default=1.0, help='gradient clipping', type=float)
    parser.add_argument('--data-path', help='data path', default='/u/songinch/song/data/speech/wsj_fbank123.h5')
    parser.add_argument('--input-dim', help='input dimension', default=123, type=int)
    parser.add_argument('--output-dim', help='output dimension', default=3436, type=int)
    parser.add_argument('--num-epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('--num-pred-layers', help='number of prediction layers between speaker embedding and scaling factor', default=1, type=int)
    parser.add_argument('--num-pred-units',
                        help='number of units in prediction layers between speaker embedding and scaling factor', default=128,
                        type=int)
    parser.add_argument('--num-tbptt-steps', default=0, help='number of truncated bptt steps', type=int)

    parser.add_argument('--num-seqsum-layers', help='number of layers in sequence summarizing neural network', default=1, type=int)
    parser.add_argument('--num-seqsum-units',  help='number of units in sequence summarizing layers', default=128,type=int)
    parser.add_argument('--seqsum-output-dim', help='output dimension of sequence summarizing neural network', default=100, type=int)

    parser.add_argument('--log-freq', help='how ferquently to display progress', default=50, type=int)
    parser.add_argument('--train-dataset', help='dataset for training', default='train_si284')
    parser.add_argument('--valid-dataset', help='dataset for validation', default='test_dev93')
    parser.add_argument('--test-dataset', help='dataset for test', default='test_eval92')

    parser.add_argument('--num-proj-units', default=0, help='number of projection units', type=int)

    parser.add_argument('--ivector-dim', help='ivector dimension', default=100, type=int)
    parser.add_argument('--use-ivector-input', help='whether to use ivectors as inputs', action='store_true')
    parser.add_argument('--use-ivector-model', help='whether to use ivectors as inputs to layers', action='store_true')
    parser.add_argument('--layer-name', help='layer name', default='IVectorLHUCLSTMLayer')
    parser.add_argument('--use-mb-loss', help='use speaker embedding loss computed over minibatch', action='store_true')  
    parser.add_argument('--mb-loss-lambda', default=0.1, help='weight for the mb-loss regularizer ', type=float)

    parser.add_argument('--reload-model', help='model path to load')
    parser.add_argument('--tmpdir', help='directory name in the /Tmp directory to save data locally', default='/Tmp/songinch/data/speech')
    parser.add_argument('--no-copy', help='do not copy data from NFS to local machine', action='store_true')
    parser.add_argument('--uni', help='make the network unidirectional', action='store_true')

    parser.add_argument('--use-layer-norm', help='whether to apply layer normalization', action='store_true')
    
def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_params(parser)
    return parser

def get_save_path(args):
    fn = os.path.splitext(os.path.basename(args.data_path))[0]
    fn = '{}_lr{}'.format(fn, args.learn_rate)
    fn = '{}_gc{}'.format(fn, args.grad_clipping)
    fn = '{}_l{}'.format(fn, args.num_layers)
    fn = '{}_n{}'.format(fn, args.num_units)
    if args.num_proj_units:
        fn = '{}_pjn{}'.format(fn, args.num_proj_units)
    if args.use_ivector_input:
        fn = '{}_iv{}'.format(fn, args.ivector_dim)
    fn = '{}_b{}'.format(fn, args.batch_size)
    if args.use_layer_norm:
        fn = '{}_ln'.format(fn)
    if args.uni:
        fn = '{}_uni'.format(fn)
    
    if args.use_ivector_model:
        fn = '{}_ivm{}'.format(fn, args.ivector_dim)

    if 'LHUC' in args.layer_name:
        fn = '{}_pdl{}_pdn{}'.format(fn, args.num_pred_layers, args.num_pred_units)
    
    if 'SeqSum' in args.layer_name:
        fn = '{}_sl{}_sn{}_so{}'.format(fn, args.num_seqsum_layers, args.num_seqsum_units, args.seqsum_output_dim)
        if args.use_mb_loss:
            fn = '{}_lambda{}'.format(fn, args.mb_loss_lambda)

    fn = '{}_{}'.format(fn, args.layer_name)

    if args.num_tbptt_steps:
        fn = '{}_tb{}'.format(fn, args.num_tbptt_steps)

    fn = '{}_{}'.format(fn, args.train_dataset)

    return fn

