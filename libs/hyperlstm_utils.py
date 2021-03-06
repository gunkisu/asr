import argparse

def add_params(parser):
    parser.add_argument('--batch-size', default=2, help='batch size', type=int)
    parser.add_argument('--num-nodes', default=10, help='number of hidden nodes', type=int)
    parser.add_argument('--num-hyper-nodes', default=4, help='number of hyper hidden nodes', type=int)
    parser.add_argument('--num-proj-nodes', default=2, help='number of proj nodes', type=int)
    parser.add_argument('--num-layers', default=1, help='number of layers', type=int)
    parser.add_argument('--learn-rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('--grad-clipping', default=1.0, help='gradient clipping', type=float)
    parser.add_argument('--data-path', help='data path', default='/u/songinch/song/data/speech/wsj_fbank123.h5')
    parser.add_argument('--input-dim', help='input dimension', default=123, type=int)
    parser.add_argument('--output-dim', help='output dimension', default=3436, type=int)
    parser.add_argument('--num-epochs', help='number of epochs', default=50, type=int)
    parser.add_argument('--train-disp-freq', help='how ferquently to display progress', default=100, type=int)
    parser.add_argument('--train-dataset', help='dataset for training', default='train_si284')
    parser.add_argument('--valid-dataset', help='dataset for validation', default='test_dev93')
    parser.add_argument('--test-dataset', help='dataset for test', default='test_eval92')

    parser.add_argument('--ivector-dim', help='ivector dimension', default=100, type=int)
    parser.add_argument('--use-ivector-input', help='whether to use ivectors as inputs', action='store_true')
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
    fn = './wsj_hyperlstm_lr{}_gc{}_l{}_n{}_b{}'.format(
            args.learn_rate, args.grad_clipping, args.num_layers, args.num_nodes, 
            args.batch_size)
    if args.use_ivector_input:
        fn = '{}_ivi{}'.format(fn, args.ivector_dim)
    if args.unidirectional:
        fn = '{}_uni'.format(fn)

    fn = '{}_hn{}_pn{}'.format(fn, args.num_hyper_nodes, args.num_proj_nodes)
    
    return fn

