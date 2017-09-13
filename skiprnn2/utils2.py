import glob
import re
import os

def find_model_iter_nums(exp_dir, best=True):
    file_pat = 'model*meta*'
    if best:
        file_pat = 'best_{}'.format(file_pat)
    model_list = glob.glob(os.path.join(exp_dir, file_pat))

    pat = re.compile('-(\d+).meta')

    tmp_model_list = []
    for model in model_list:
        m = pat.search(model)
        if m:
            tmp_model_list.append(int(m.group(1)))

    tmp_model_list.sort(reverse=True)
    return tmp_model_list

