import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, random_split

from Trainer import Trainer
# from trainer_2 import Trainer
from options import MonodepthOptions
from data_loaders import Prepare_dataset, Prepare_dataset_test

options = MonodepthOptions()
opts = options.parse()

save_folder = 'D:\\depth_estimation_implementation\\rough_2\\visual_saves'

data_path = 'D:\\depth_estimation_implementation\\rough\\Final_prepared_data.npy'
total_data = np.load(data_path, allow_pickle = True)
random.shuffle(total_data)
print('Total KITTI data shape: ', total_data.shape)

train_data = total_data[0:39996] # 0:39996
validation_data = total_data[40000:42460] # final data shape -> 42,467
# validation_data = total_data[40000:40024]

appolo_path = 'D:\\depth_estimation_implementation\\Appoloscapes_dataset.npy'
total_appolo_data = np.load(appolo_path, allow_pickle = True)
print('Total Appolo dataset: ', total_appolo_data.shape)
total_appolo_data = total_appolo_data[0:63708]

Appolo_flag = True

train_set = Prepare_dataset(train_data, total_appolo_data, appolo = Appolo_flag)
val_set = Prepare_dataset_test(validation_data)#, total_appolo_data, appolo = False)

train_loader = DataLoader(train_set, batch_size = 12, shuffle = True, pin_memory = True, num_workers = 6)
val_loader = DataLoader(val_set, batch_size = 12, shuffle = False, pin_memory = True, num_workers = 6)

if __name__ == '__main__':
    trainer = Trainer(opts, train_loader, val_loader, save_folder, batch_size = 12, use_affinity = Appolo_flag)
    
    trainer.train()
    # trainer.val()
    
    # trainer.test_model()
