
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal


def l2_norm(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        m = normal.Normal(0, 1)
        s = m.sample(x.shape)

        r = torch.rand(x.shape).sub(0.5).to(x.device)
        r = l2_norm(r)
        for number in range(self.vat_iter):
            r.requires_grad_(True)
            
            adv_exp = x + self.xi * r
            adv_pred = F.log_softmax(model(adv_exp), dim=1)
            adv_dist = F.kl_div(adv_pred, pred, reduction="mean")
            adv_dist.backward()
            r = l2_norm(r.grad.data)
            model.zero_grad()

        r_adv = r * self.eps
        adv_inp = x + r_adv
        adv_pred = F.log_softmax(model(adv_inp), dim=1)
        vt_loss = F.kl_div(adv_pred, pred, reduction="mean")

        return vt_loss,adv_inp
