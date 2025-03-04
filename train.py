import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, random_split

# from Trainer import Trainer
# from Trainer_distilation import Trainer
from Trainer_distillation_CRF import Trainer
# from Trainer_distillation_refinement import Trainer
# from Train_distillation_new import Trainer
# from train_controlled_diffusion import Trainer
from options import MonodepthOptions
from data_loaders import Prepare_dataset, Prepare_dataset_test

CUDA_LAUNCH_BLOCKING = 1

options = MonodepthOptions()
opts = options.parse()

save_folder = 'D:\\depth_estimation_implementation\\Visual_saves\\zxzxzxzx'

# data_path = 'D:\\depth_estimation_implementation\\rough\\Final_prepared_data.npy'
data_path = 'D:\\depth_estimation_implementation\\rough_3\\Final_prepared_data_second.npy'
total_data = np.load(data_path, allow_pickle = True)
# random.shuffle(total_data)
print('Total KITTI data shape: ', total_data.shape)

train_data = total_data[0:39996] # 9996, 39996
validation_data = total_data[40000:42460] # final data shape -> 42,467

video_1 = validation_data[0:451] #451
video_2 = validation_data[453:1621]
video_3 = validation_data[1623:2459]

# validation_data = total_data[40000:40024]
# np.save('complete_validation_KITTI_dataset.npy', validation_data)
# np.save('small_KITTI_training_dataset.npy', train_data)
# np.save('KITTI_validation_data.npy', validation_data)
# np.save('KITTI_training_data.npy', train_data)

appolo_path = 'D:\\depth_estimation_implementation\\Appoloscapes_dataset.npy'
total_appolo_data = np.load(appolo_path, allow_pickle = True)
print('Total Appolo dataset: ', total_appolo_data.shape)

# random.shuffle(total_appolo_data)

total_appolo_data = total_appolo_data[0:39984] #63708 #39984
np.random.shuffle(total_appolo_data)

# np.save('small_Appollo_training_dataset.npy', total_appolo_data)

Appolo_flag = False

# validation_data = validation_data[0:12]

train_set = Prepare_dataset(train_data, total_appolo_data, appolo = Appolo_flag)
val_set_1 = Prepare_dataset_test(video_1)#, total_appolo_data, appolo = False)
val_set_2 = Prepare_dataset_test(video_2)
val_set_3 = Prepare_dataset_test(video_3)

train_loader = DataLoader(train_set, batch_size = 4, shuffle = True, pin_memory = True, num_workers = 3)
val_loader_1 = DataLoader(val_set_1, batch_size = 4, shuffle = False, pin_memory = True, num_workers = 3)
val_loader_2 = DataLoader(val_set_2, batch_size = 4, shuffle = False, pin_memory = True, num_workers = 3)
val_loader_3 = DataLoader(val_set_3, batch_size = 4, shuffle = False, pin_memory = True, num_workers = 3)

loaders = [val_loader_1, val_loader_2, val_loader_3]

if __name__ == '__main__':
    trainer = Trainer(opts, train_loader, loaders, save_folder, batch_size = 4, use_affinity = Appolo_flag)
    
    # trainer.pretrain_affinity()
    trainer.train()
    
    # trainer.val()
    # trainer.test_model() # batch in the testing loop is set to 1
    # trainer.study_model()
    
    