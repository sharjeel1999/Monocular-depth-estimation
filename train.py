import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, random_split

# from Trainer import Trainer
from Trainer_distilation import Trainer
from options import MonodepthOptions
from data_loaders import Prepare_dataset, Prepare_dataset_test

CUDA_LAUNCH_BLOCKING = 1

options = MonodepthOptions()
opts = options.parse()

save_folder = 'D:\\depth_estimation_implementation\\Visual_saves\\finetuned_A3'

data_path = 'D:\\depth_estimation_implementation\\rough\\Final_prepared_data.npy'
total_data = np.load(data_path, allow_pickle = True)
# random.shuffle(total_data)
print('Total KITTI data shape: ', total_data.shape)

train_data = total_data[0:39996] # 9996, 39996
validation_data = total_data[40000:42460] # final data shape -> 42,467
# validation_data = total_data[40000:40024]
# np.save('complete_validation_KITTI_dataset.npy', validation_data)
# np.save('small_KITTI_training_dataset.npy', train_data)
# np.save('KITTI_validation_data.npy', validation_data)
# np.save('KITTI_training_data.npy', train_data)

appolo_path = 'D:\\depth_estimation_implementation\\Appoloscapes_dataset.npy'
total_appolo_data = np.load(appolo_path, allow_pickle = True)
print('Total Appolo dataset: ', total_appolo_data.shape)

random.shuffle(total_appolo_data)

total_appolo_data = total_appolo_data[0:39984] #63708 #39984
np.random.shuffle(total_appolo_data)

# np.save('small_Appollo_training_dataset.npy', total_appolo_data)

Appolo_flag = False

# validation_data = validation_data[0:12]

train_set = Prepare_dataset(train_data, total_appolo_data, appolo = Appolo_flag)
val_set = Prepare_dataset_test(validation_data)#, total_appolo_data, appolo = False)

train_loader = DataLoader(train_set, batch_size = 12, shuffle = True, pin_memory = True, num_workers = 3)
val_loader = DataLoader(val_set, batch_size = 12, shuffle = False, pin_memory = True, num_workers = 3)

if __name__ == '__main__':
    trainer = Trainer(opts, train_loader, val_loader, save_folder, batch_size = 12, use_affinity = Appolo_flag)
    
    # trainer.pretrain_affinity()
    # trainer.train()
    # trainer.val()
    trainer.test_model()
    # trainer.study_model()

