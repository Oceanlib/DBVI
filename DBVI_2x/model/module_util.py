from torch import nn
from torch.nn import init


def randomInitNet(net_l, iniType='kaiming', scale=1.0):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if any([isinstance(m, nn.Conv2d), isinstance(m, nn.ConvTranspose2d), isinstance(m, nn.Linear)]):
                if iniType == 'normal':
                    init.normal_(m.weight, 0.0, 0.2)
                elif iniType == 'xavier':
                    init.xavier_normal_(m.weight, gain=0.2)
                elif iniType == 'kaiming':
                    init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                elif iniType == 'orthogonal':
                    init.orthogonal_(m.weight, gain=0.2)
                elif iniType == 'default':
                    pass

                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
                m.weight.data *= scale
            elif any([isinstance(m, nn.InstanceNorm2d), isinstance(m, nn.LocalResponseNorm),
                      isinstance(m, nn.BatchNorm2d), isinstance(m, nn.GroupNorm)]):
                try:
                    init.constant_(m.weight, 1.0)
                    init.constant_(m.bias, 0.0)
                except Exception as e:
                    pass
