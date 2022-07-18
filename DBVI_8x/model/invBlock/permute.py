import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class TransConv1x1(nn.Module):
    def __init__(self, inCh):
        super(TransConv1x1, self).__init__()
        self.w_shape = [inCh, inCh]
        w_init = np.linalg.qr(np.random.randn(*self.w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))

    def get_weight(self, x, rev):
        b, c, h, w = x.shape
        dlogdet = torch.slogdet(self.weight)[1] * h * w  # slogdet(A) = torch.log(torch.abs(torch.det(A)))

        if not rev:
            weight = self.weight
        else:
            weight = self.weight.t()

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, x, logdet=None, rev=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        logdet = 0.0 if logdet is None else logdet
        weight, dlogdet = self.get_weight(x, rev)
        z = F.conv2d(x, weight)
        if not rev:
            logdet = logdet + dlogdet
        else:
            logdet = logdet - dlogdet

        return z, logdet


class InvConv1x1(nn.Module):
    def __init__(self, inCh):
        super(InvConv1x1, self).__init__()
        self.w_shape = [inCh, inCh]
        w_init = np.linalg.qr(np.random.randn(*self.w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))

    def get_weight(self, x, rev):
        b, c, h, w = x.shape
        dlogdet = torch.slogdet(self.weight)[1] * h * w  # slogdet(A) = torch.log(torch.abs(torch.det(A)))

        if not rev:
            weight = self.weight
        else:
            weight = torch.inverse(self.weight)

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, x, logdet=None, rev=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        logdet = 0.0 if logdet is None else logdet
        weight, dlogdet = self.get_weight(x, rev)
        z = F.conv2d(x, weight)
        if not rev:
            logdet = logdet + dlogdet
        else:
            logdet = logdet - dlogdet

        return z, logdet


class Permute2d(nn.Module):
    def __init__(self, inCh, shuffle=True):
        super().__init__()
        self.inCh = inCh
        # self.indices = torch.arange(self.inCh - 1, -1, -1, dtype=torch.long)
        # self.indices_inverse = torch.zeros(self.inCh, dtype=torch.long)
        self.register_buffer('indices', torch.arange(self.inCh - 1, -1, -1, dtype=torch.long))
        self.register_buffer('indices_inverse', torch.zeros(self.inCh, dtype=torch.long))

        for i in range(self.inCh):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.inCh):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, x, logdet, rev=False):
        assert len(x.size()) == 4

        if not rev:
            x = x[:, self.indices, :, :]
        else:
            x = x[:, self.indices_inverse, :, :]

        return x, logdet


class InvConvLU1x1(nn.Module):  # is not recommended
    def __init__(self, inCh):
        super(InvConvLU1x1, self).__init__()
        w_shape = [inCh, inCh]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
        s = torch.diag(upper)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        upper = torch.triu(upper, 1)
        l_mask = torch.tril(torch.ones(w_shape), -1)
        eye = torch.eye(*w_shape)

        self.register_buffer('p', p)  # .cuda() will work only on register_buffer
        self.register_buffer('sign_s', sign_s)
        self.register_buffer('l_mask', l_mask)
        self.register_buffer('eye', eye)

        self.lower = nn.Parameter(lower)
        self.log_s = nn.Parameter(log_s)
        self.upper = nn.Parameter(upper)

        self.w_shape = w_shape

    def get_weight(self, x, rev):
        b, c, h, w = x.shape

        lower = self.lower * self.l_mask + self.eye

        u = self.upper * self.l_mask.transpose(0, 1).contiguous()
        u += torch.diag(self.sign_s * torch.exp(self.log_s))

        dlogdet = torch.sum(self.log_s) * h * w
        if not rev:
            weight = torch.matmul(self.p, torch.matmul(lower, u))
        else:
            u_inv = torch.inverse(u)
            l_inv = torch.inverse(lower)
            p_inv = torch.inverse(self.p)

            weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, x, logdet=None, rev=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(x, rev)
        z = F.conv2d(x, weight)

        if not rev:
            logdet = logdet + dlogdet
        else:
            logdet = logdet - dlogdet
        return z, logdet