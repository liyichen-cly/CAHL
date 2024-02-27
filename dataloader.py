from torch.utils.data import Dataset

class EData(Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, index):
		eid = self.data[index][0]
		label = self.data[index][1]
		return eid, label

	def __len__(self):
		return len(self.data)

