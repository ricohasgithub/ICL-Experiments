def _linear_warmup_and_sqrt_decay(self, step, warmup_steps=1000, lr_max=3e-4):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    else:
        return lr_max * (warmup_steps ** 0.5) / (step ** 0.5)
    
# plot learning rate schedule
import matplotlib.pyplot as plt
import numpy as np

steps = np.arange(0, 10000, 100)
lrs = [_linear_warmup_and_sqrt_decay(None, step) for step in steps]
plt.plot(steps, lrs)
plt.xlabel('Step')
plt.ylabel('Learning rate')
plt.title('Learning rate schedule')
plt.show()
