import torch
import torchvision
import os
import torch.utils
import torch.utils.data

path2train_dir = "./images/src_train"
path2test_dir = "./images/src_test"
path2train_label = "./images/tgt_train"
path2test_label = "./images/tgt_test"
fname_train = os.listdir(path2train_dir)
fname_test = os.listdir(path2test_dir)
fname_train.sort()
fname_test.sort()
fname_train_label = os.listdir(path2train_label)
fname_train_label.sort()
fname_test_label = os.listdir(path2test_label)
fname_test_label.sort()
print(f"the len of fname_train is {len(fname_train)}")

def get_data_tensor(path,filenames):
    data_in_tensor = []
    for fname in filenames:
        data_in_tensor.append(
            torchvision.io.read_image(
                os.path.join(path,fname)
            )
        )
    return data_in_tensor

train_features = get_data_tensor(path2train_dir,fname_train)
train_label = get_data_tensor(path2train_label,fname_train_label)
test_features = get_data_tensor(path2test_dir,fname_test)
test_labels = get_data_tensor(path2test_label,fname_test_label)
print(f"the len of train features is {len(train_features)}")


class ImageData(torch.utils.data.Dataset):
    def __init__(self,features,labels):
        super().__init__()
        self.features = features
        self.label = labels
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def normalize(self,img):
        return self.transform(img.float())

    def __getitem__(self, index):
        return self.normalize(self.features[index]),self.label[index]

    def __len__(self):
        return len(self.features)

def get_train_and_test_data(train_features=train_features,train_label=train_label,
                            test_features=test_features,test_label=test_labels):
    return ImageData(train_features,train_label),ImageData(test_features,test_label)
    
     