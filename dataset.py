import torch
from torch.utils.data import Dataset


class SherlockDataset(Dataset):
    def __init__(self, granularite, predict=False):
        self.predict = predict
        self.name = 'sherlock_' + granularite
        data_file = 'hochelaga_{}.pt'.format({'el': 'electeur', 'post': 'postal'}[granularite])
        self.predictors, self.targets, self.key_cols = torch.load(data_file)
        self.num_features = self.predictors.size(1)
        self.num_classes = self.targets.size(1)
    
    def __getitem__(self, index):
        if self.predict:
            return self.predictors[index], self.targets[index], self.key_cols[index]
        else:
            return self.predictors[index], self.targets[index]
    
    def __len__(self):
        return self.predictors.size(0)
