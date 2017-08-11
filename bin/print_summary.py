from __future__ import print_function
import tensorflow as tf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('event_file')
    parser.add_argument('summary_kind')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--min', action="store_true", help="Print only the minimum value")
    group.add_argument('--max', action="store_true", help="Print only the maximum value")
    args = parser.parse_args()

    history = []
    for e in tf.train.summary_iterator(args.event_file):
        for v in e.summary.value:
            if v.tag == args.summary_kind:
                history.append(float(v.simple_value))
    
    if args.max or args.min:
        history.sort()

    if args.max: 
        print(history[-1])
    elif args.min:
        print(history[0])
    else:
        for h in history: print(h)
    
