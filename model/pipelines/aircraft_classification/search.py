import math
from collections import OrderedDict
from typing import Dict

from hyperopt import hp


def make_space_dict() -> Dict[str, OrderedDict]:
    return {
        'resnet50_narrow': OrderedDict([
            ('lrA', hp.uniform('lrA', 2e-4, 4e-4)),
            ('wdA', hp.loguniform('wdA', math.log(1e-2), math.log(1))),
            ('lrB', hp.uniform('lrB', 2e-4, 4e-4)),
            ('wdB', hp.loguniform('wdB', math.log(1e-2), math.log(1))),
            ('lr_t0', hp.choice('lr_t0', [10])),
            ('lr_f', hp.choice('lr_f', [2.0])),
            ('lr_warmup', hp.choice('lr_warmup', [3])),
            ('lb_smooth', hp.loguniform('lb_smooth', math.log(1e-4), math.log(1))),
            ('cutmix_alpha', hp.loguniform('cutmix_alpha', math.log(1e-2), math.log(1))),
            ('mixup_alpha', hp.loguniform('mixup_alpha', math.log(1e-2), math.log(1)))
        ]),
        'resnet50_wide': OrderedDict([
            ('lrA', hp.loguniform('lrA', math.log(1e-4), math.log(1e-2))),
            ('wdA', hp.loguniform('wdA', math.log(1e-3), math.log(1e+1))),
            ('lrB', hp.loguniform('lrB', math.log(1e-4), math.log(1e-2))),
            ('wdB', hp.loguniform('wdB', math.log(1e-3), math.log(1e+1))),
            ('lr_t0', hp.choice('lr_t0', [10])),
            ('lr_f', hp.choice('lr_f', [2.0])),
            ('lr_warmup', hp.choice('lr_warmup', [3])),
            ('lb_smooth', hp.loguniform('lb_smooth', math.log(1e-5), math.log(1))),
            ('cutmix_alpha', hp.loguniform('cutmix_alpha', math.log(1e-3), math.log(1e+1))),
            ('mixup_alpha', hp.loguniform('mixup_alpha', math.log(1e-3), math.log(1e+1)))
        ])
    }
