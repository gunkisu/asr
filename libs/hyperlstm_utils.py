import argparse

def add_params(parser):
    parser.add_argument('--batch-size', default=1, help='batch size', type=int)
    parser.add_argument('--num-nodes', default=64, help='number of hidden nodes', type=int)
    parser.add_argument('--num-hyper-nodes', default=32, help='number of hyper hidden nodes', type=int)
    parser.add_argument('--num-proj-nodes', default=32, help='number of proj nodes', type=int)
    parser.add_argument('--num-layers', default=1, help='number of layers', type=int)
    parser.add_argument('--learn-rate', default=0.0001, help='learning rate', type=float)
    parser.add_argument('--grad-clipping', default=1.0, help='gradient clipping', type=float)
    parser.add_argument('--data-path', help='data path', default='/u/songinch/song/data/speech/wsj_fbank123.h5')
    parser.add_argument('--input-dim', help='input dimension', default=123, type=int)
    parser.add_argument('--output-dim', help='output dimension', default=3436, type=int)
    parser.add_argument('--num-epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('--num-hyperlstm-layers', help='number of hyperlstm layers', default=1, type=int)

    parser.add_argument('--train-disp-freq', help='how ferquently to display progress', default=100, type=int)
    parser.add_argument('--updater', help='sgd or momentum', default='momentum')
    parser.add_argument('--train-dataset', help='dataset for training', default='train_si84')
    parser.add_argument('--valid-dataset', help='dataset for validation', default='test_dev93')
    parser.add_argument('--test-dataset', help='dataset for test', default='test_eval92')

    parser.add_argument('--truncate-ivectors', help='truncate ivectors', action='store_true')
    parser.add_argument('--ivector-dim', help='ivector dimension', default=100, type=int)
    parser.add_argument('--use-ivectors', help='whether to use ivectors', action='store_true')
    parser.add_argument('--lhuc', help='whether to use lhuc', action='store_true')
    parser.add_argument('--tied-lhuc', help='whether to use tied lhuc', action='store_true')

    parser.add_argument('--reload-model', help='model path to load')
    parser.add_argument('--tmpdir', help='directory name in the /Tmp directory to save data locally', default='/Tmp/songinch/data/speech')
    parser.add_argument('--no-copy', help='do not copy data from NFS to local machine', action='store_true')
    parser.add_argument('--unidirectional', help='make the network unidirectional', action='store_true')

    
def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_params(parser)
    return parser

def get_save_path(args):
    fn = './wsj_hyperlstm_lr{}_gc{}_nl{}_nn{}_b{}'.format(
            args.learn_rate, args.grad_clipping, args.num_layers, args.num_nodes, 
            args.batch_size)
    if args.use_ivectors:
        fn = '{}_iv{}'.format(fn, args.ivector_dim)
    if args.unidirectional:
        fn = '{}_uni'.format(fn)

    fn = '{}_hl{}'.format(fn, args.num_hyperlstm_layers)

    if not args.tied_lhuc:
        fn = '{}_hnn{}_pnn{}'.format(args.num_hyper_nodes, args.num_proj_nodes)
  
    if args.lhuc:
        fn = '{}_lhuc'.format(fn)
    
    
    if args.tied_lhuc:
        fn = '{}_tied_lhuc'.format(fn)
    return fn

