import tensorflow as tf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('event_file')
    parser.add_argument('summary_kind')

    args = parser.parse_args()
    for e in tf.train.summary_iterator(args.event_file):
        for v in e.summary.value:
            if v.tag == args.summary_kind:
                print(v.simple_value)
            