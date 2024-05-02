from PIL import Image
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        self.images = []
        self.labels = []
        self.transform = transforms.ToTensor()
        self.complte_path = dataset_path + "/labels.csv"
        with open(self.complte_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")

            next(reader)

            for row in reader:
                self.images.append(dataset_path + "/" + row[0])
                self.labels.append(LABEL_NAMES.index(row[1]))
        

    def __len__(self):
        """
        Your code here
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        image = Image.open(self.images[idx])
        image_tensor = self.transform(image)
        return image_tensor, self.labels[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
