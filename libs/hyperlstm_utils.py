import argparse

def add_params(parser):
    parser.add_argument('--batch-size', default=2, help='batch size', type=int)
    parser.add_argument('--num-nodes', default=10, help='number of hidden nodes', type=int)
    parser.add_argument('--num-hyper-nodes', default=4, help='number of hyper hidden nodes', type=int)
    parser.add_argument('--num-proj-nodes', default=2, help='number of proj nodes', type=int)
    parser.add_argument('--num-layers', default=1, help='number of layers', type=int)
    parser.add_argument('--learn-rate', default=0.0001, help='learning rate', type=float)
    parser.add_argument('--grad-clipping', default=1.0, help='gradient clipping', type=float)
    parser.add_argument('--data-path', help='data path', default='/u/songinch/song/data/speech/wsj_fbank123.h5')
    parser.add_argument('--input-dim', help='input dimension', default=123, type=int)
    parser.add_argument('--output-dim', help='output dimension', default=3436, type=int)
    parser.add_argument('--num-epochs', help='number of epochs', default=50, type=int)
#    parser.add_argument('--num-hyperlstm-layers', help='number of hyperlstm layers', default=1, type=int)
    parser.add_argument('--num-pred-layers', help='number of prediction layers between speaker embedding and scaling factor', default=1, type=int)
    parser.add_argument('--num-pred-nodes',
                        help='number of units in prediction layers between speaker embedding and scaling factor', default=100,
                        type=int)

    parser.add_argument('--num-seqsum-layers', help='number of layers in sequence summarizing neural network', default=2, type=int)
    parser.add_argument('--num-seqsum-nodes',  help='number of units in sequence summarizing layers', default=512,type=int)
    parser.add_argument('--seqsum-output-dim', help='output dimension of sequence summarizing neural network', default=100, type=int)

    parser.add_argument('--train-disp-freq', help='how ferquently to display progress', default=100, type=int)
    parser.add_argument('--updater', help='one of [sgd|momentum|adam]', default='adam')
    parser.add_argument('--train-dataset', help='dataset for training', default='train_si284')
    parser.add_argument('--valid-dataset', help='dataset for validation', default='test_dev93')
    parser.add_argument('--test-dataset', help='dataset for test', default='test_eval92')

    parser.add_argument('--reparam', help='function for reparametrisation', default='relu')
    parser.add_argument('--pred-act', help='activation function for the prediction network', default='relu')

    parser.add_argument('--ivector-dim', help='ivector dimension', default=100, type=int)
    parser.add_argument('--use-ivector-input', help='whether to use ivectors as inputs', action='store_true')
    parser.add_argument('--use-ivector-model', help='whether to use ivectors as inputs to layers', action='store_true')
    parser.add_argument('--layer-name', help='layer name', default='IVectorLHUCLSTMLayer')
    parser.add_argument('--use-mb-loss', help='use speaker embedding loss computed over minibatch', action='store_true')  
    parser.add_argument('--mb-loss-lambda', default=0.1, help='weight for the mb-loss regularizer ', type=float)

    parser.add_argument('--reload-model', help='model path to load')
    parser.add_argument('--tmpdir', help='directory name in the /Tmp directory to save data locally', default='/Tmp/songinch/data/speech')
    parser.add_argument('--no-copy', help='do not copy data from NFS to local machine', action='store_true')
    parser.add_argument('--unidirectional', help='make the network unidirectional', action='store_true')

    parser.add_argument('--use-layer-norm', help='whether to apply layer normalization', action='store_true')
    
def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_params(parser)
    return parser

def get_save_path(args):
    fn = './wsj_hyperlstm_lr{}_gc{}_nl{}_nn{}_b{}'.format(
            args.learn_rate, args.grad_clipping, args.num_layers, args.num_nodes, 
            args.batch_size)
    if args.use_ivector_input:
        fn = '{}_ivi{}'.format(fn, args.ivector_dim)
    if args.use_ivector_model:
        fn = '{}_ivm{}'.format(fn, args.ivector_dim)
    if args.unidirectional:
        fn = '{}_uni'.format(fn)

#    fn = '{}_hl{}'.format(fn, args.num_hyperlstm_layers)

    if 'Hyper' in args.layer_name: 
        fn = '{}_hnn{}_pnn{}'.format(fn, args.num_hyper_nodes, args.num_proj_nodes)
    
    if 'LHUC' in args.layer_name:
        fn = '{}_rp{}'.format(fn, args.reparam)
        fn = '{}_npl{}_npn{}'.format(fn, args.num_pred_layers, args.num_pred_nodes)
        fn = '{}_a{}'.format(fn, args.pred_act)
    
    if 'SeqSum' in args.layer_name:
        fn = '{}_nsl{}_nsn{}_sod{}'.format(fn, args.num_seqsum_layers, args.num_seqsum_nodes, args.seqsum_output_dim)
        if args.use_mb_loss:
            fn = '{}_lambda{}'.format(fn, args.mb_loss_lambda)

    if args.use_layer_norm:
        fn = '{}_ln'.format(fn)

    fn = '{}_{}'.format(fn, args.layer_name)
  
    return fn

