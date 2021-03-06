import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import SherlockDataset
from utils import MetricHistory, SimpleProfiler
from visdom_helper import VisdomHelper


# TODO
# par electeur
    # par electeur: test set bias
    # grid search
# Dropout?
# depth/width
# Sparsify

def str2bool(x):
    if x == 1 or x == 'Y' or x == 'T':
        return True
    elif x == 0 or x == 'N' or x == 'F':
        return False
    else:
        raise TypeError('Could not cast to bool.')

parser = argparse.ArgumentParser()
parser.add_argument('granularite', choices=['el', 'post'])
parser.add_argument('--pred', action='store_true', help='Generate predictions')
# Training setup and hyper params
parser.add_argument('--load', '-l', type=str, default=None)
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
parser.add_argument('--steps', default='100,150', type=str, help='epoch ids of lr step, e.g. 30,60,100')
parser.add_argument('--reg', default=2e-4, type=float, help='l2 regul')
parser.add_argument('--n_epochs', '-e', default=300, type=int, help='max epochs')
parser.add_argument('--batch_size', '-b', default=32, type=int)
# Utilities
parser.add_argument('--tuning', action='store_true')
parser.add_argument('--devices', '-d', type=str, default=None, help='device ids, e.g. 0,1,3')
parser.add_argument('--workers', '-w', type=int, default=1, help='number of DataLoader workers')
parser.add_argument('--cut', action='store_true', help='cut epoch short')
parser.add_argument('--delete', type=str2bool, default=True, help='delete parent checkpoint')
parser.add_argument('--novis', action='store_true', help='turn off visdom')
parser.add_argument('--print_freq', default=-1, type=int)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()


#torch.backends.cudnn.benchmark = True  # Improves speed


class ClassificationTraining(object):
    last_checkpoint = None

    def __init__(self):
        self.monitors = None
        self.err_monitors = None
        self.rerr_monitors = None
        self.class_names = ['qs', 'pq', 'plq', 'caq']  # , 'on', 'pv']
        self.parent = ''
        self.start_epoch = 0
        self.checkpoint_filename_pattern = None
        self.lr_steps = [] if args.steps is None else [int(x) for x in args.steps.split(',')]
        self.scheduler = None

        self.dataset = None
        self.train_transform = None
        self.val_transform = None
        self.train_loader = None
        self.val_loader = None
        self.num_classes = None

        self.net = None
        self.sparse_conv, self.sparse_linear = [], []
        self.total_sparse_conv, self.total_sparse_linear = 0, 0
        self.box_encoder = None
        self.optimizer = None
        self.vis = None  # for visdom

        self.prepare_data()
        self.prepare_model()
        self.prepare_visdom()

    def make_monitors(self):
        self.monitors = {name: MetricHistory() for name in ('loss_train', 'loss_val', 'accu_train', 'accu_val',
                                                            'alive_conv', 'alive_lin')}
        self.err_monitors = [MetricHistory() for _ in self.class_names]
        self.rerr_monitors = [MetricHistory() for _ in self.class_names]

    @staticmethod
    def get_validation_set(dataset, validation_ratio=0.1):
        filename = dataset.name + '_valsplit.pkl'
        try:
            train_idx, val_idx = torch.load(filename)
        except FileNotFoundError:
            print('WARNING: Generating new validation set')
            num_samples = len(dataset)
            num_val = int(num_samples * validation_ratio)
            shuffled_idx = torch.randperm(num_samples).long()
            train_idx = shuffled_idx[num_val:]
            val_idx = shuffled_idx[:num_val]
            torch.save((train_idx, val_idx), filename)
        return train_idx, val_idx

    def prepare_data(self):
        print('==> Preparing data...')

        self.dataset = SherlockDataset(args.granularite, predict=args.pred)

        if args.pred:
            self.val_loader = data.DataLoader(self.dataset, args.batch_size, num_workers=args.workers, pin_memory=True)
        else:
            train_idx, val_idx = self.get_validation_set(self.dataset, validation_ratio=0.1)
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            self.train_loader = data.DataLoader(self.dataset, args.batch_size, num_workers=args.workers,
                                                sampler=train_sampler, pin_memory=True)
            self.val_loader = data.DataLoader(self.dataset, args.batch_size, num_workers=args.workers,
                                              sampler=val_sampler, pin_memory=True)

        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes

    def make_optimizer(self, l2_reg=args.reg):
        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=l2_reg)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.lr_steps, gamma=0.1)

    def set_l2_reg(self, amount):
        self.make_optimizer(l2_reg=amount)

    def prepare_model(self):
        print('==> Preparing model...')

        dropout_p = 0.2

        # Build network
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            #torch.nn.Dropout(dropout_p),
            torch.nn.Linear(64, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            #torch.nn.Dropout(dropout_p),
            torch.nn.Linear(256, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            #torch.nn.Dropout(dropout_p),
            torch.nn.Linear(1024, self.num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

        if args.cuda:
            self.net.cuda()

        self.make_optimizer()
        self.make_monitors()

        # Load model state, optimizer state and monitors
        if not args.tuning and args.path is not None:
            list_of_files = glob.glob(os.path.join(args.path, '*.pt'))
            if len(list_of_files) > 0:
                last_file = max(list_of_files, key=os.path.getctime)
                last_file = os.path.join(args.path, os.path.basename(last_file))  # glob *sometimes* return abs path
                self.load(last_file)
            else:
                print('WARNING: Checkpoint dir is empty. Training from scratch.')
        elif args.load is not None:
            self.load(args.load)

        self.criterion = torch.nn.KLDivLoss()
        if args.cuda:
            self.criterion.cuda()

    def prepare_visdom(self):
        env_name = 'sherlock'
        if args.novis:
            self.vis = VisdomHelper.make_dummy()
            args.print_freq = 30 if args.print_freq == -1 else args.print_freq
        else:
            self.vis = VisdomHelper(self.monitors, env_name)

    @staticmethod
    def full_path(filename):
        if args.path is not None:
            path = args.path
        else:
            path = os.environ.get('SHERLOCK_CKPT', 'checkpoint')
        if not os.path.isdir(path):
            os.makedirs(path)
        return os.path.join(path, filename)

    def load(self, filename):
        """Load a checkpoint of the training state. See save() for format."""

        c = torch.load(filename)

        if type(c) is dict:
            sd = c['state_dict']
            self.net.load_state_dict(sd)
            if 'monitors' in c:  # Remove the branching eventually
                self.monitors = c['monitors']
            else:
                self.monitors = {'loss_train': c['train_monitor'], 'loss_val': c['val_monitor'],
                                 'accu_train': MetricHistory(), 'accu_val': MetricHistory()}
            if 'optimizer' in c:  # Remove the branching eventually
                self.optimizer.load_state_dict(c['optimizer'])
        else:
            raise RuntimeError('Unsupported checkpoint. (Not a dict)')

        self.parent = filename
        self.last_checkpoint = filename
        self.start_epoch = self.monitors['loss_train'].num_epochs

    def save(self):
        """Save a checkpoint of the training state.

        Saved fields:
            state_dict      state_dict from PyTorch model in self.net (collections.OrderedDict)
            monitors        Dict of MetricHistory
            parent          filename of the parent checkpoint (str)
        """

        pattern = '{}_{}_{}ep.pt' if self.checkpoint_filename_pattern is None else self.checkpoint_filename_pattern
        filename = pattern.format('sherlock1', time.strftime("%Y-%m-%d_%H-%M-%S"),
                                  self.monitors['loss_train'].num_epochs)
        full_filename = self.full_path(filename)
        c = {
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitors': self.monitors,
            'parent': self.parent,
            'args': vars(args)  # convert args to dict
        }
        torch.save(c, full_filename)
        if not args.tuning and args.delete and self.last_checkpoint is not None:
            os.remove(self.last_checkpoint)
        self.last_checkpoint = full_filename
        return filename

    def error_fn(self, pred, target, relative=False):
        if relative:
            e = torch.abs(pred - target) / target
        else:
            e = torch.abs(pred - target)
        return torch.mean(e, dim=0)

    def update_err_monitors(self, err, rerr):
        for mon, cerr in zip(self.err_monitors, err):
            mon.update(float(cerr))
        for mon, cerr in zip(self.rerr_monitors, rerr):
            mon.update(float(cerr))

    def epoch_end_err_monitors(self):
        for mon in self.err_monitors + self.rerr_monitors:
            mon.end_epoch()

    def train_epoch(self, epoch):
        #print('\nEpoch: %d' % epoch)
        self.net.train()

        profiler = SimpleProfiler(['loader', 'variables', 'forward', 'backward', 'sync', 'monitor'])

        for batch_idx, (input_data, targets) in enumerate(self.train_loader):
            if args.cut and batch_idx == 20:
                break

            profiler.step()  # loader

            if args.cuda:
                input_data = input_data.cuda(async=True)
                targets = targets.cuda(async=True)

            input_var = torch.autograd.Variable(input_data)
            targets_norm = targets / targets.sum(dim=1, keepdim=True)  # sum to 1
            target_var = torch.autograd.Variable(targets_norm)
            profiler.step()  # variables

            output = self.net(input_var)
            loss = self.criterion(output, target_var)
            profiler.step()   # forward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            profiler.step()  # backward

            loss_value = float(loss)
            profiler.step()  # sync (fetching the value of the loss causes a sync between CPU and GPU)

            # compute % (ajusted for total pct)
            total_pct = targets.sum(dim=1, keepdim=True)
            pred = torch.exp(output.data) * total_pct

            err = self.error_fn(pred, targets)
            self.monitors['accu_train'].update(float(err[0]))  # qs
            self.monitors['loss_train'].update(loss_value)

            if args.print_freq != -1 and batch_idx % args.print_freq == 0:
                print('{}     {}/{}    Last loss: {:.6f}    Avg loss: {:.6f}'.format(time.strftime("%Y-%m-%d %H-%M-%S"),
                                                                                     batch_idx, len(self.train_loader),
                                                                                     loss_value,
                                                                                     self.monitors['loss_train'].avg))

            self.vis.update_status(epoch, 'Training', batch_idx, len(self.train_loader), loss_value,
                                   self.monitors['loss_train'].avg, self.monitors['accu_train'].avg)
            if batch_idx > 0:
                self.vis.show_dict(profiler.to_dict(), win='time')
                if batch_idx < 2000 and batch_idx % 200 == 0:
                    self.vis.loss(self.monitors['loss_train'].history, 'Loss Per Batch', 'Iterations', 'loss_batch')
            profiler.step()  # monitor

        self.monitors['loss_train'].end_epoch()
        self.monitors['accu_train'].end_epoch()

    def val_epoch(self, epoch):
        #print('\nTest: %d' % epoch)
        self.net.eval()

        for batch_idx, (input_data, targets) in enumerate(self.val_loader):
            if args.cut and batch_idx == 20:
                break

            if args.cuda:
                input_data = input_data.cuda(async=True)
                targets = targets.cuda(async=True)

            input_var = torch.autograd.Variable(input_data, volatile=True)
            targets_norm = targets / targets.sum(dim=1, keepdim=True)  # sum to 1
            target_var = torch.autograd.Variable(targets_norm, volatile=True)

            # compute output
            output = self.net(input_var)
            loss = self.criterion(output, target_var)
            loss_value = float(loss)

            # compute % (ajusted for total pct)
            total_pct = targets.sum(dim=1, keepdim=True)
            pred = torch.exp(output.data) * total_pct

            err = self.error_fn(pred, targets)
            self.monitors['accu_val'].update(float(err[0]))  # qs
            self.monitors['loss_val'].update(loss_value)
            rerr = self.error_fn(output.data, targets, relative=True)
            self.update_err_monitors(err, rerr)

            if args.print_freq != -1 and batch_idx % args.print_freq == 0:
                print('{}     {}/{}    Last loss: {:.6f}    Avg loss: {:.6f}'.format(time.strftime("%Y-%m-%d %H-%M-%S"),
                                                                                     batch_idx, len(self.train_loader),
                                                                                     float(loss),
                                                                                     self.monitors['loss_val'].avg))

            self.vis.update_status(epoch, 'Test', batch_idx, len(self.val_loader), float(loss),
                                   self.monitors['loss_val'].avg, self.monitors['accu_val'].avg)

        self.monitors['loss_val'].end_epoch()
        self.monitors['accu_val'].end_epoch()
        self.epoch_end_err_monitors()
        
    def predict(self):
        print('==> Predicting...')
        self.net.eval()

        predictions = []
        ids = []
        
        for batch_idx, (input_data, targets, key_cols) in enumerate(self.val_loader):
            if args.cuda:
                input_data = input_data.cuda(async=True)
                targets = targets.cuda(async=True)

            input_var = torch.autograd.Variable(input_data, volatile=True)
            targets_norm = targets / targets.sum(dim=1, keepdim=True)  # sum to 1
            target_var = torch.autograd.Variable(targets_norm, volatile=True)

            # compute output
            output = self.net(input_var)
            loss = self.criterion(output, target_var)
            loss_value = float(loss)

            # compute % (ajusted for total pct)
            total_pct = targets.sum(dim=1, keepdim=True)
            pred = torch.exp(output.data) * total_pct
            
            # store predictions
            predictions.append(pred)
            ids.append(key_cols)

            err = self.error_fn(pred, targets)
            self.monitors['accu_val'].update(float(err[0]))  # qs
            self.monitors['loss_val'].update(loss_value)
            rerr = self.error_fn(output.data, targets, relative=True)
            self.update_err_monitors(err, rerr)

            if args.print_freq != -1 and batch_idx % args.print_freq == 0:
                print('{}     {}/{}    Last loss: {:.6f}    Avg loss: {:.6f}'.format(time.strftime("%Y-%m-%d %H-%M-%S"),
                                                                                     batch_idx, len(self.train_loader),
                                                                                     float(loss),
                                                                                     self.monitors['loss_val'].avg))

            self.vis.update_status(0, 'Predict', batch_idx, len(self.val_loader), float(loss),
                                   self.monitors['loss_val'].avg, self.monitors['accu_val'].avg)

        # Save predictions
        with open('pred.csv', 'w') as f:
            for batch_ids, batch_pred in zip(ids, predictions):
                post_codes, section_ids = batch_ids
                for postc, sect, pred in zip(post_codes, section_ids, batch_pred):
                    line = [postc, str(sect)] + ['{:.3f}'.format(p) for p in pred]
                    line_str = ','.join(line)
                    f.write(line_str + '\n')

    def train(self, num_epochs=200, do_save=True):
        print('==> Training...')
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            self.vis.plot_metrics()
            self.vis.plot_monitors(self.err_monitors, 'Erreur absolue', 'Erreur absolue', self.class_names)
            self.vis.plot_monitors(self.rerr_monitors, 'Erreur relative', 'Erreur relative', self.class_names)
            self.scheduler.step(epoch)
            self.train_epoch(epoch)
            self.val_epoch(epoch)
            if do_save:
                self.save()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def print_args(args_dict):
    print('=== ARGS ===================================')
    keys = set(args_dict.keys()) - {'cut', 'gates', 'novis', 'delete', 'onlyviz'}
    kcolwidth = max(len(k) for k in keys)
    for k in sorted(keys):
        print(('{:<' + str(kcolwidth) + '} {}').format(k, args_dict[k]))
    print('============================================')


def hyperparam_tuning():
    import itertools

    # post            0.2     5e-4
    # el-nodup(fk)    0.2     1e-3
    # el-nodup        0.2     2e-4
    lr_choices = [5e-2, 1e-1, 2e-1, 5e-1]
    reg_choices = [1e-4, 2e-4, 5e-4, 1e-3, 5e-3]

    results = {}

    try:
        for lr, reg in itertools.product(lr_choices, reg_choices):
            print('Trying: {:.2e}  {:.2e}'.format(lr, reg))
            args.lr = lr
            args.reg = reg

            training = ClassificationTraining()
            training.train(args.n_epochs, do_save=False)

            lasterr = np.mean(training.monitors['accu_val'].epochs[:-4])
            print('Last err: {}'.format(lasterr))

            fn = training.save()
            print('Saved: ' + fn)
            results[lr, reg] = lasterr, fn
    except KeyboardInterrupt:
        pass

    torch.save(results, 'tuning.pt')


if __name__ == '__main__':

    print_args(vars(args))

    if args.tuning:
        hyperparam_tuning()
    elif args.pred:
        training = ClassificationTraining()
        training.predict()
    else:
        training = ClassificationTraining()
        training.train(args.n_epochs)
