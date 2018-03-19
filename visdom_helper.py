from collections import OrderedDict
import numpy as np
import visdom

_vis = visdom.Visdom()


class VisdomHelper(object):
    def __init__(self, monitors, env_name='pytorch', dummy=False):
        if dummy or not _vis.check_connection():
            if not dummy:
                print('ERROR: Could not connect to visdom server! Did you start it?\n    python -m visdom.server')
            self.die()
            return

        self.env = env_name
        self.monitors = monitors
        _vis.close(env=self.env)
        _vis.text('Initializing...', env=self.env, win='status')

    @staticmethod
    def make_dummy():
        print('WARNING: visdom is OFF')
        return VisdomHelper(None, dummy=True)

    def show_dict(self, data, win):
        html = '<style>.k{margin-right:1em}</style><table>'
        for key, value in data.items():
            html += '<tr><td class="k">' + key + '</td><td>' + str(value) + '</td></tr>'
        html += '</table>'
        _vis.text(html, env=self.env, win=win)

    def update_status(self, epoch, phase, batch_idx, total_batches, loss, avg_loss, avg_accu):
        self.show_dict(OrderedDict([
            ('Epoch', epoch),
            ('Phase', phase),
            ('Batch', '{}/{}    {:.0f}%'.format(batch_idx, total_batches, batch_idx / total_batches * 100)),
            ('Loss', '{:.6f}'.format(loss)),
            ('Avg loss', '{:.6f}'.format(avg_loss)),
            ('Avg accu', '{:.6f}'.format(avg_accu))
        ]), win='status')

    def loss(self, loss_list, title, xlabel, win):
        n_hidden_iter = 20
        if len(loss_list) <= n_hidden_iter:
            pass
        else:
            x = np.arange(len(loss_list))[n_hidden_iter:]
            y = np.array(loss_list)[n_hidden_iter:]
            _vis.line(y, x, env=self.env, win=win,
                      opts={'title': title, 'xlabel': xlabel, 'ylabel': 'Loss'})

    def loss_train_val(self):
        if self.monitors['loss_val'].num_epochs == 0:
            return
        Y = np.stack([self.monitors['loss_train'].epochs, self.monitors['loss_val'].epochs], axis=1)
        X = np.arange(self.monitors['loss_val'].num_epochs)
        _vis.line(Y, X, env=self.env, win='loss',
                  opts=dict(title='Loss', xlabel='Epoch', ylabel='Loss', legend=['Train', 'Test']))

    def accu_train_val(self):
        if self.monitors['accu_val'].num_epochs == 0:
            return
        Y = np.stack([self.monitors['accu_train'].epochs, self.monitors['accu_val'].epochs], axis=1)
        X = np.arange(self.monitors['accu_val'].num_epochs)
        _vis.line(Y, X, env=self.env, win='accu',
                  opts=dict(title='Accuracy', xlabel='Epoch', ylabel='Accuracy', legend=['Train', 'Test']))

    def plot_compression(self):
        if self.monitors['alive_conv'].num_epochs == 0:
            return
        Y = np.stack([self.monitors['alive_conv'].epochs, self.monitors['alive_lin'].epochs], axis=1)
        X = np.arange(self.monitors['alive_conv'].num_epochs)
        _vis.line(Y, X, env=self.env, win='alive',
                  opts=dict(title='Alive neurons', xlabel='Epoch', ylabel='Alive ratio', legend=['Conv', 'FC']))

    def plot_metrics(self):
        self.loss_train_val()
        self.accu_train_val()
        self.plot_compression()

    def image(self, img, win):
        _vis.image(img, env=self.env, win=win)

    def die(self):
        """This method transform the instance into a potato"""

        def nop(*args, **kwargs):
            pass

        for m in dir(self):
            if '__' not in m:
                setattr(self, m, nop)
