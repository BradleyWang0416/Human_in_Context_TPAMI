import torch
import numpy as np


def get_targets(target_type='default'):
    if target_type == 'default':
        targets = []
        x_coors = np.linspace(-0.5, 0.5, 4)
        y_coors = np.linspace(-0.5, 0.5, 4)
        z_coors = np.linspace(-0.5, 0.5, 4)
        for x in x_coors:
            for y in y_coors:
                for z in z_coors:
                    targets.append(torch.tensor([x, y, z]))
        targets = torch.stack(targets[:60])   # (60, 3)
    else:
        raise NotImplementedError
    if torch.cuda.is_available():
        targets = targets.float().cuda()
    return targets