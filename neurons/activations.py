import torch


def f1(x, deriv=False):
    if not deriv:
        return torch.where(x > 0, 0.5 * torch.log(torch.square(x) + 1), 0.0)
    return torch.where(x > 0, x / (1 + torch.square(x)), 0.0)
    
def f2(x, deriv=False):
    if not deriv:
        return torch.where(x > 0, 2 / torch.pi * torch.arctan(x), 0.0)
    return torch.where(x > 0, 2 / torch.pi / (1 + torch.square(x)), 0.0)
    
def f3(x, deriv=False):
    absx = 1 + torch.abs(x)
    if not deriv:
        return torch.maximum(x / absx, 0.0)
    return torch.heaviside(x, 0) / absx
    
def f4(x, deriv=False):
    if not deriv:
        return torch.maximum(torch.tanh(x), 0.0)
    return torch.heaviside(x, 0.0) / torch.square(torch.cosh(x))
    
def f5(x, deriv=False, device="cpu"):
    squ = torch.square(x)
    if not deriv:
        return (squ / (squ + 1.0)).where(x > 0.0, torch.zeros(x.shape, device=device))
    return (2.0 * x / torch.square(1.0 + squ)).where(x > 0.0, torch.zeros(x.shape, device=device))
    
def exp1(x, deriv=False):
    if not deriv:
        return torch.where(x > 0.0, 1.0 - torch.exp(-x), 0.0)
    return torch.where(x > 0.0, torch.exp(-x), 0.0)

def sigmoid(x, deriv=False):
    if not deriv:
        y = torch.where(x >= 0.0, torch.exp(-x), torch.exp(x))
        return y / (1.0 + y)
    else:
        return sigmoid(x) * (1.0 - sigmoid(x))

