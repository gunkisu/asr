#!/usr/bin/env python
from __future__ import print_function

import time

from libs.deep_lstm_utils import *

from libs.utils import StopWatch, Rsync
import data.wsj.fuel_utils as fuel_utils

from fuel.transformers import MultiProcessing

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    print(args)

    rsync = Rsync(args.tmpdir)
    rsync.sync(args.data_path)
    args.data_path = os.path.join(args.tmpdir, os.path.basename(args.data_path))

    train_ds = fuel_utils.get_datastream(path=args.data_path,
                                  which_set=args.train_dataset,
                                  batch_size=args.batch_size, 
                                  use_ivectors=args.use_ivectors, 
                                  truncate_ivectors=args.truncate_ivectors, 
                                  ivector_dim=args.ivector_dim)
    train_ds = MultiProcessing(train_ds, max_store=200)
    valid_ds = fuel_utils.get_datastream(path=args.data_path,
                                  which_set=args.valid_dataset,
                                  batch_size=args.batch_size, 
                                  use_ivectors=args.use_ivectors,
                                  truncate_ivectors=args.truncate_ivectors,
                                  ivector_dim=args.ivector_dim)
    test_ds = fuel_utils.get_datastream(path=args.data_path,
                                  which_set=args.test_dataset,
                                  batch_size=args.batch_size, 
                                  use_ivectors=args.use_ivectors,
                                  truncate_ivectors=args.truncate_ivectors,
                                  ivector_dim=args.ivector_dim)
    

    for e_idx in range(1, args.num_epochs+1):
        sw = StopWatch()
        
        for b_idx, data in enumerate(train_ds.get_epoch_iterator(), start=1):
            sw.print_elapsed()

            time.sleep(0.1)    
            sw.reset()
        
