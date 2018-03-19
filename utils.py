import timeit
from collections import OrderedDict, deque
import numpy as np


class LossHistory(object):
    """History of the loss during training. (Lighter version of MetricHistory)

    Usage:
        monitor = LossHistory()
        ...
        # Call update at each iteration
        monitor.update(2.3)
        ...
        monitor.avg  # returns the average loss
        ...
        monitor.end_epoch()  # call at epoch end
        ...
        monitor.epochs  # returns the loss curve as a list
    """

    def __init__(self):
        self.history = []
        self.epochs = []
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_iter = 0
        self.num_epochs = 0

    def update(self, value):
        self.history.append(value)
        self.sum += value
        self.count += 1
        self._avg = self.sum / self.count
        self.num_iter += 1

    @property
    def avg(self):
        return self._avg

    def end_epoch(self):
        self.epochs.append(self._avg)
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_epochs += 1


MetricHistory = LossHistory


class SimpleProfiler(object):
    """Measures the time taken by each step of a loop.

    Usage:
        prof = PoorMansProfiler(['data_load', 'forward', 'backward'])
        for i in range(num_batches):
            x, y = load_batch(i)
            prof.step()  # data_load done
            pred_y = model(x)
            loss = criterion(y, pred_y)
            prof.step()  # forward done
            loss.backward()
            optimizer.step()
            prof.step()  # backward done
        profiling_result = prof.to_dict()
    """

    def __init__(self, step_names):
        self.step_names = step_names
        self.cur_idx = 0
        self.last_time = timeit.default_timer()
        self.avglen = 20
        self.times = [deque(maxlen=self.avglen) for _ in range(len(step_names))]

    def step(self):
        steptime = timeit.default_timer() - self.last_time
        self.times[self.cur_idx].append(steptime)
        self.last_time = timeit.default_timer()
        self.cur_idx = (self.cur_idx + 1) % len(self.step_names)

    def __getitem__(self, idx):
        """Get the average time of a step"""
        return sum(self.times[idx]) / self.avglen

    def to_dict(self):
        d = OrderedDict()
        for i, name in enumerate(self.step_names):
            d[name] = '{:.5f}'.format(self[i])
        return d


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    buf = np.moveaxis(buf, -1, 0)
    return buf


def plot_gate_histogram(conv_gates, linear_gates):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    def per_layer_histogram(gates):
        l_hist = []
        for l in gates:
            hist, _ = np.histogram(l, bins=10, range=(0, 1))
            l_hist.append(hist)
        return np.array(l_hist)

    binleft = np.linspace(0.0, 0.9, num=10)
    xtickslabels = ['{:.1f}-{:.1f}'.format(x, x + 0.1) for x in binleft]

    conv_hist = per_layer_histogram(conv_gates)
    lin_hist = per_layer_histogram(linear_gates)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 5))
    for ax, img, title in zip(axes.flat, [conv_hist, lin_hist], ['Conv layers', 'FC layers']):
        ax.set_title(title)
        im = ax.imshow(img, norm=colors.LogNorm(), aspect='auto')
        ax.set_ylabel('Layer idx')
        if title == 'Conv layers':
            ax.get_xaxis().set_visible(False)
        else:
            ax.set_xlabel('Histogram of gate values')
            ax.set_xticklabels(xtickslabels)
        ax.set_yticks(np.arange(img.shape[0]))
        ax.set_xticks(np.arange(img.shape[1]))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    img = fig2data(fig)
    #plt.close()  Intermittent crashes if used
    return img


def plot_num_alive(conv_gates):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    all_gates = conv_gates
    alive = [np.count_nonzero(g) for g in all_gates]
    total = [len(g) for g in all_gates]
    values_2d = np.stack([alive, total], axis=0)
    sorted_idx = np.argsort(total)
    values_2d = values_2d[:, sorted_idx]
    plt.figure(figsize=(7, 5))
    width = 0.4
    left1 = np.arange(len(alive))
    left2 = left1 + width
    plt.bar(left2, values_2d[1], width, label='Total')
    plt.bar(left1, values_2d[0], width, label='Alive')
    plt.xlabel('Layer idx')
    plt.legend()
    plt.grid()

    img = fig2data(plt.gcf())
    #plt.close()  Intermittent crashes if used
    return img
