import pickle
from torch.utils.data import Dataset


def load_pheme_pkl():
    with open('data/PHEME.pkl', 'rb') as file_handler:
        data_list = pickle.load(file_handler)
        return data_list

dataset = load_pheme_pkl()
def from_label_to_index(data):
    #print("------")
    def _helper(_tensor):
        unique = {}
        counter = 0
        for row in _tensor:
            for col in row:
                if col.item() in unique.keys():
                    continue
                else: 
                    unique[col.item()] = counter
                    counter += 1

        return unique

    def _replacer(_tensor, unique):
        i = 0
        j = 0
        #print(_tensor)
        for row in _tensor: 
            for col in row:
                if col.item() in unique.keys():
                    _tensor[i][j] = unique[col.item()]
                j += 1
            j = 0
            i += 1 
        #print(_tensor)
    edge_index = _helper(data.edge_index)
    _replacer(data.edge_index, edge_index)
    return data

for data in dataset:
    data = from_label_to_index(data)

class PHEME_Dataset(Dataset):
    def __init__(self):
        self.indices = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]