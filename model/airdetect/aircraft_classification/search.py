import math
from collections import OrderedDict
from typing import Dict

from hyperopt import hp


def make_space_dict() -> Dict[str, OrderedDict]:
    return {
        'resnet50': OrderedDict([
            ('lrA', hp.uniform('lrA', 1e-4, 4e-4)),
            ('wdA', hp.loguniform('wdA', math.log(1e-2), math.log(1))),
            ('lrB', hp.uniform('lrB', 1e-4, 4e-4)),
            ('wdB', hp.loguniform('wdB', math.log(1e-2), math.log(1))),
            ('nn_frz', hp.choice('nn_frz', [4])),
            ('lr_t0', hp.choice('lr_t0', [10])),
            ('lr_f', hp.choice('lr_f', [2])),
            ('lr_w', hp.choice('lr_w', [3])),
            ('x_lbs', hp.loguniform('x_lbs', math.log(1e-4), math.log(1e-1))),
            ('x_cut', hp.choice('x_cut', [0, 1])),
            ('x_cut_a', hp.loguniform('x_cut_a', math.log(1e-2), math.log(1))),
            ('x_mxp', hp.choice('x_mxp', [0, 1])),
            ('x_mxp_a', hp.loguniform('x_mxp_a', math.log(1e-2), math.log(1)))
        ]),

        'resnet50_cutmix': OrderedDict([
            ('lrA', hp.uniform('lrA', 1e-4, 4e-4)),
            ('wdA', hp.loguniform('wdA', math.log(1e-2), math.log(1))),
            ('lrB', hp.uniform('lrB', 1e-4, 4e-4)),
            ('wdB', hp.loguniform('wdB', math.log(1e-2), math.log(1))),
            ('nn_frz', hp.choice('nn_frz', [4])),
            ('lr_t0', hp.choice('lr_t0', [10])),
            ('lr_f', hp.choice('lr_f', [2])),
            ('lr_w', hp.choice('lr_w', [3])),
            ('x_lbs', hp.loguniform('x_lbs', math.log(1e-4), math.log(1e-1))),
            ('x_cut', hp.choice('x_cut', [1])),
            ('x_cut_a', hp.uniform('x_cut_a', 5e-2, 20e-2))
        ])
    }
