
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