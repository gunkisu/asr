import argparse
import os

from fuel.server import start_server

from data.wsj import fuel_utils
from libs.utils import StopWatch, Rsync


def create_data_stream(args):
    print(args)
    sw = StopWatch()

    with sw:
        print('Copying data to local machine...')
        rsync = Rsync(args.tmpdir)
        rsync.sync(args.data_path)

    args.data_path = os.path.join(args.tmpdir, os.path.basename(args.data_path))
    return fuel_utils.get_datastream(path=args.data_path,
                                      which_set=args.dataset,
                                      batch_size=args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', help='data path', default='/u/songinch/song/data/speech/wsj_fbank123.h5')
    parser.add_argument('--dataset', help='dataset for training', default='train_si84_rand')
    parser.add_argument('--batch-size', default=1, help='batch size', type=int)
    parser.add_argument('--tmpdir', help='directory name in the /Tmp directory to save data locally', default='/Tmp/songinch/data/speech')
    parser.add_argument('--port', help='port number', default=5557, type=int)
    args = parser.parse_args()
    start_server(create_data_stream(args), port=args.port)
