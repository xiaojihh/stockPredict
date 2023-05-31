from torch.utils.data import Dataset



class TrainSet(Dataset):
    def __init__(self, data, pred_days=30):
        self.pred_days = pred_days
        self.data, self.label = data[:, :-self.pred_days].float(), data[:, -self.pred_days:].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)