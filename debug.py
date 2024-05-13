from matplotlib import pyplot as plt
from models.utils_quant import SymQuantizer, AsymQuantizer
import torch


x = torch.linspace(0, 10, 1000)
x.requires_grad = True

num_bits = 4
max_input = 10
s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)

y1 = torch.round(x * s) / s
y2 = AsymQuantizer.apply(x, torch.tensor([-10, 10]), num_bits, False)

plt.plot(x.detach().numpy(), y2.detach().numpy(), label='quant')


# plot d quant / d x
grad = torch.autograd.grad(y2, x, torch.ones_like(x))[0]

plt.plot(x.detach().numpy(), grad.detach().numpy(), label='grad')
plt.legend()
plt.savefig('debug_grad.png')
