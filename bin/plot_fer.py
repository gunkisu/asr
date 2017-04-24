# IPython log file
import argparse
import matplotlib.pyplot as plt
import numpy

def uniq(seq):
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('logfile', help='log file')
args = parser.parse_args()
print args

with open(args.logfile) as f:
    lines_to_plot = [l for l in f if 'Valid' in l]
    lines_to_plot = uniq(lines_to_plot)
    nums_to_plot = [float(l.strip().split()[-1]) for l in lines_to_plot] 
    plt.plot(nums_to_plot)
    plt.show()

