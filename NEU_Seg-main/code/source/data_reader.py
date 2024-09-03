import torch
import torchvision
import os
import torch.utils
import torch.utils.data

path2train_dir = "../images/src_training"
path2test_dir = "../images/src_test"
path2train_label = "../images/tgt_training"
path2test_label = "../images/tgt_test"
fname_train = os.listdir(path2train_dir)
fname_test = os.listdir(path2test_dir)
fname_train.sort()
fname_test.sort()
fname_train_label = os.listdir(path2train_label)
fname_train_label.sort()
fname_test_label = os.listdir(path2test_label)
fname_test_label.sort()
print(f"the len of fname_train is {len(fname_train)}")


def get_data_tensor(path, filenames):
    data_in_tensor = []
    for fname in filenames:
        data_in_tensor.append(
            torchvision.io.read_image(
                os.path.join(path, fname)
            )
        )
    return data_in_tensor


train_features = get_data_tensor(path2train_dir, fname_train)
train_labels = get_data_tensor(path2train_label, fname_train_label)
test_features = get_data_tensor(path2test_dir, fname_test)
test_labels = get_data_tensor(path2test_label, fname_test_label)
print(f"the len of train features is {len(train_features)}")


class ImageData(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.label = labels
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def normalize(self, img):
        return self.transform(img.float() / 255)

    def __getitem__(self, index):
        return self.normalize(self.features[index]), self.label[index]

    def __len__(self):
        return len(self.features)



def load_data(batch_size):
    train_iter = torch.utils.data.DataLoader(
        ImageData(train_features, train_labels),
        batch_size, shuffle=True, drop_last=True, num_workers=4
    )

    test_iter = torch.utils.data.DataLoader(
        ImageData(test_features, test_labels),
        drop_last=True, num_workers=4
    )

    return train_iter, test_iter
