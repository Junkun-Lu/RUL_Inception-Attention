from torch.utils.data import Dataset

#  ---------------------------- Dataset ----------------------------------
class FEMTOData(Dataset):

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        sample_y = self.data_y[index]

        return sample_x, sample_y

    def __len__(self):
        return len(self.data_x)