import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

WIDTH, HEIGHT = 32, 32
batch_size = 32


class EmojiDataset(Dataset):
    def __init__(self, csv_files, image_folder, transform=None):
        self.dataframe = pd.concat([pd.read_csv(csv_file)
                                   for csv_file in csv_files])
        self.images_folder = image_folder
        self.dataframe['image_path'] = self.dataframe['file_name'].str.replace(
            '\\', '/')
        self.image_paths = self.dataframe['image_path'].tolist()
        self.titles = self.dataframe['prompt'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.images_folder + '/' + self.image_paths[idx]
        title = self.titles[idx]
        title = title.replace('"', '').replace("'", '')
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, title


transform = transforms.Compose([
    transforms.Resize(
        (WIDTH, HEIGHT),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
