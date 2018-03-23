import matplotlib.pyplot as plt
import torch
import numpy as np

results = torch.load('tuning.pt')


def get_data():
    lr = []
    reg = []
    err = []
    for p, res in results.items():
        ierr, filename = res
        print(p, filename)
        ilr, ireg = p
        lr.append(ilr)
        reg.append(ireg)
        err.append(ierr)
    return lr, reg, err


def plot(lr, reg, err ):
    plt.close()
    plt.figure()
    get_data()
    plt.scatter(np.log10(lr), np.log10(reg), c=err)
    plt.xlabel('lr')
    plt.ylabel('reg')
    plt.colorbar()
    plt.show()


lr, reg, err = get_data()
plot(lr, reg, err)
