import numpy as np

def set_model_param_value(params_var_list, params_val_list):
    if len(params_var_list) != len(params_val_list):
        raise ValueError("mismatch: got %d values to set %d parameters" %
                         (len(params_val_list), len(params_var_list)))

    for p, v in zip(params_var_list, params_val_list):
        if p.get_value().shape != v.shape:
            raise ValueError(
                "mismatch: parameter has shape %r but value to "
                "set has shape %r" %
                (p.get_value().shape, v.shape))
        else:
            p.set_value(v)

def get_model_param_values(params_list):
    return [p.get_value() for p in params_list]

def get_update_params_values(params_var_dict):
    params_val_dict = {}
    for name, var in params_var_dict.iteritems():
        params_val_dict[name] = var.get_value()

def get_layer_param_dict(layer, trainable=True, param_dict=None):
    param_list = layer.get_params(trainable=trainable)
    if param_dict is None:
        param_dict = {}
    for p in param_list:
        param_dict[p.name.split('.')[-1]] = p
    return param_dict

def count_params(model_params):
    shapes = [p.get_value().shape for p in model_params]
    counts = [np.prod(shape) for shape in shapes]
    return sum(counts)