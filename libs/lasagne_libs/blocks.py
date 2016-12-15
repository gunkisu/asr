import numpy
from lasagne import init, nonlinearities
from lasagne.layers import Gate, ConcatLayer
from libs.lasagne_libs.layers import LSTMLayer


def BidirLSTMBlock(data_layer,
                   mask_layer=None,
                   num_units=None,
                   ingate=Gate(),
                   forgetgate=Gate(),
                   cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                   outgate=Gate(),
                   nonlinearity=nonlinearities.tanh,
                   cell_init=init.Constant(0.),
                   hid_init=init.Constant(0.),
                   dropout_ratio=0.2,
                   learn_init=False,
                   peepholes=True,
                   gradient_steps=-1,
                   grad_clipping=0,
                   unroll_scan=False,
                   precompute_input=True,
                   only_return_final=False,
                   **kwargs):
    if num_units is None:
        num_units = numpy.prod(data_layer.output_shape[2:])

    forward_lstm_layer = LSTMLayer(incoming=data_layer,
                                   mask_input=mask_layer,
                                   num_units=num_units,
                                   ingate=ingate,
                                   forgetgate=forgetgate,
                                   cell=cell,
                                   outgate=outgate,
                                   nonlinearity=nonlinearity,
                                   cell_init=cell_init,
                                   hid_init=hid_init,
                                   dropout_ratio=dropout_ratio,
                                   backwards=False,
                                   learn_init=learn_init,
                                   peepholes=peepholes,
                                   gradient_steps=gradient_steps,
                                   grad_clipping=grad_clipping,
                                   unroll_scan=unroll_scan,
                                   precompute_input=precompute_input,
                                   only_return_final=only_return_final,
                                   **kwargs)

    backward_lstm_layer = LSTMLayer(incoming=data_layer,
                                    mask_input=mask_layer,
                                    num_units=num_units,
                                    ingate=ingate,
                                    forgetgate=forgetgate,
                                    cell=cell,
                                    outgate=outgate,
                                    nonlinearity=nonlinearity,
                                    cell_init=cell_init,
                                    hid_init=hid_init,
                                    dropout_ratio=dropout_ratio,
                                    backwards=True,
                                    learn_init=learn_init,
                                    peepholes=peepholes,
                                    gradient_steps=gradient_steps,
                                    grad_clipping=grad_clipping,
                                    unroll_scan=unroll_scan,
                                    precompute_input=precompute_input,
                                    only_return_final=only_return_final,
                                    **kwargs)

    output_layer = ConcatLayer(incomings=[forward_lstm_layer,
                                          backward_lstm_layer],
                               axis=-1)

    return output_layer
