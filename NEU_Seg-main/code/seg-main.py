import torch
import torchvision
import os
import torch.utils
import torch.utils.data
import data_reader as dr



def main():
    train_iter, test_iter = dr.load_data(64)
    for feature, label in train_iter:
        print(feature)
        break



if __name__ == '__main__':
    main()