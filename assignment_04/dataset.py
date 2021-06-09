import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import os.path
from my_utils import device

N_OFFLINE_AUGMENTATIONS = 6 # the number of offline data-augmentations + 1
                            # (for no augmentation)

BATCH_SIZE = 20
    
onlineAugmentations = [
    lambda x: x, # the identity transformation
    transforms.Compose([transforms.CenterCrop(size=(200,200)),
                       transforms.Resize((256, 256))]), # Crop to the center
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                                       hue=0.05), # color jitter
    transforms.Compose([transforms.Pad(50),
                       transforms.Resize((256, 256))]), # add padding
]

class PersonRobotDataset(torch.utils.data.Dataset):
    """A dataset of 120 photographs of persons and 120
    photographs of humandoid robots.
    Data augmentations are used to increase
    the number of samples.
    """

    def __init__(self, train_val_test):
        """The value of train_val_test should be set to
        0 to get the training dataset,
        1 to get the validation data set or
        2 to get the test data set."""

        if int(train_val_test) == 0:
            set_type = 'train'
        elif int(train_val_test) == 1:
            set_type = 'val'
        elif int(train_val_test) == 2:
            set_type = 'test'
        else:
            raise ValueError(f"train_val_test must be 0,1 or 2 but equals {train_val_test}")
        
        possible_indeces = [42,  74,  84,  12, 113,  29,  55,  18, 105,  69,  34,  87,   5,
                            85,  58,   3,   9,  65, 111,  15,  76, 100,  52,  79, 114,  75,
                           102,  51,  43,  20,  94,  57,  16,  60, 104,  37, 119,  92, 103,
                            11,  96,  80, 118,  23,  14,  56,  81, 107,  68,  90,  54, 109,
                            53,  40,  44,  27,  32, 110,  22,  73,  72,  93,  82,  39,  19,
                            17,  45, 120,  35,  59,  83,  46,  28,  31,  88,  36,  61,  33,
                           101,  38,  41,  66,  71,  89,  30,   2,  25,  10,   8,  47, 108,
                            63, 116,  67,  48,   1,  78,  50,  97,  64,   6,  91, 106,  77,
                            21,  62,  70, 117,  26,  13,  24,  99,  95,  86,   7, 115,   4,
                           112,  49,  98] # a random permutation of range(1, 121)

        if train_val_test == 0:
            # set indeces to be 78 from 120 possible indeces (65%)
            self.indeces = possible_indeces[:78]
        elif train_val_test == 1:
            # set indeces to be 18 from 120 possible indeces (15%)
            self.indeces = possible_indeces[78:96]
        elif train_val_test == 2:
            # set indeces to be 24 from 120 possible indeces (20%)
            self.indeces = possible_indeces[96:]
            
        dataset_file = f"dataset/processed_dataset_{set_type}.pt"
        if os.path.exists(dataset_file):
            self.robots, self.persons = torch.load(dataset_file, map_location=device)
            return

        datasets_shape = (len(self.indeces,), N_OFFLINE_AUGMENTATIONS,
                         3, 256, 256)
        self.persons= torch.zeros(datasets_shape).to(device)
        self.robots = torch.zeros(datasets_shape).to(device)
        compose = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((256, 256))])
        transform = lambda x: compose(x)[:3] #.movedim((0,1,2), (2,0,1))

        progress_bar = tqdm(enumerate(self.indeces), total=len(self.indeces))

        for data_index, picture_index in progress_bar:
            img = Image.open(f'dataset/person/person{picture_index}.png')
            self.persons[data_index, 0] = transform(img)

            img = Image.open(f'dataset/robot/robot{picture_index}.png')
            self.robots[data_index, 0] = transform(img)

            for aug in range(1, N_OFFLINE_AUGMENTATIONS):
                for label in ['person', 'robot']:
                    file_name = f"dataset/{label}/{label}{picture_index}_aug{aug}.png"
                    img = Image.open(file_name)

                    if label == 'person':
                        self.persons[data_index, aug] = transform(img)
                    else:
                        self.robots[data_index, aug] = transform(img)
        
        torch.save((self.persons, self.robots), dataset_file)

    def __getitem__(self, idx):
        # determine whether we use a person or robot picture
        if idx%2 == 0:
            dataset = self.persons
            label = 0.0
        else:
            dataset = self.robots
            label = 1.0
        idx = idx//2

        # determine the online augmentation we should use
        n_online_augment = len(onlineAugmentations)
        online_augment = onlineAugmentations[idx%n_online_augment]
        idx = idx//n_online_augment

        # determine the offline augmenation we should use
        offline_augment = idx%N_OFFLINE_AUGMENTATIONS

        idx = idx//N_OFFLINE_AUGMENTATIONS

        # determine the image id
        image_id = idx

        return online_augment(dataset[image_id, offline_augment]), label

    def __len__(self):
        return len(self.indeces) * N_OFFLINE_AUGMENTATIONS \
               * len(onlineAugmentations) * 2


# the data loaders
train_loader = torch.utils.data.DataLoader(dataset=PersonRobotDataset(0), batch_size=BATCH_SIZE, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=PersonRobotDataset(1), batch_size=BATCH_SIZE, 
                                         shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=PersonRobotDataset(2), batch_size=BATCH_SIZE, 
                                          shuffle=True)