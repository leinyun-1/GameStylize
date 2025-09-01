import json
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import cv2
from PIL import Image

import h5py

class DoomH5Dataset(Dataset):
    def __init__(self,root='/mnt/kaiwu-user-carlfang/doom_dataset/freedoom_random_128_u2_rs_fix_v6_1280_720_hdf5',size=[64,360,640],prefix_action=2):
        super().__init__()
        self.root = root
        self.frames = size[0]
        self.height = size[1]
        self.width = size[2]
        self.action_num = 9
        self.prefix_action = prefix_action
        #self.subs = os.listdir(root)
        meta_file_path = root + '.txt'
        with open(meta_file_path, 'r', encoding='utf-8') as f:
            self.subs = [line.strip() for line in f if line.strip()]
        print("dataset size: ",len(self.subs))
        self.frame_process = transforms.Compose([
            #transforms.CenterCrop(size=(self.height, self.width)),
            transforms.Resize(size=(self.height, self.width), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def img_process(self,img):
        img = Image.fromarray(img)
        img = self.frame_process(img)
        return img 
    
    def get_action_onehot(self,game_actions_idxs,st_frame):
        # frames = len(game_actions_idxs)
        # action_onehot = np.zeros((frames,self.action_num))
        # action_onehot[[i for i in range(frames)], np.array(game_actions_idxs).astype(np.int32)] = 1. # f * 9
        # return action_onehot

        init_action = np.zeros(self.action_num)
        game_actions = []
        pre_actions = []
        if self.prefix_action > 1:
            for i in range(st_frame - self.prefix_action + 1,st_frame):
                pre_actions.append(init_action.copy())
                if i < 0:
                    pre_actions[-1][self.action_num-1] = 1
                else:
                    pre_actions[-1][round(game_actions_idxs[i])] = 1
            pre_actions = np.concatenate(pre_actions)
        for i in range(self.frames):
            game_actions.append(init_action.copy())
            game_actions[-1][round(game_actions_idxs[st_frame+i])] = 1
            if self.prefix_action > 1:
                pre_actions = np.concatenate([pre_actions, game_actions[-1]])
                game_actions[-1] = pre_actions.copy()
                pre_actions = pre_actions[self.action_num:]
        game_actions = np.stack(game_actions) # nf, 9*(prefix+1)
        return game_actions


    
    def read_h5(self,filename):
        with h5py.File(filename, "r") as f:
            img_actions = f['images'][:]
            metas = f['metas'][:]

        # print(img_actions.shape) # [n_frames, 720 * 1280 * 3 + 1]
        # print(metas.shape) # [n_frames, 33]

        name = filename.split('/')[-1].split('.')[0]

        video = []
        actions = []
        for step in range(img_actions.shape[0]):
            img = img_actions[step][:720 * 1280 * 3]
            img = img.astype(np.uint8)
            img = img.reshape(720, 1280, 3)
            img = self.img_process(img)
            video.append(img)
            action = int(img_actions[step][-1])
            actions.append(action)

        video = torch.stack(video,dim=1) #  3 f h w 
        actions = np.array(actions) # f 
        

        target_frames = self.frames
        start_frame_id = np.random.randint(0,video.shape[1]-target_frames)
        video = video[:,start_frame_id:start_frame_id+target_frames] # 3 f h w 
        #actions = actions[start_frame_id:start_frame_id+target_frames]
        actions = self.get_action_onehot(actions,start_frame_id) # f * 9(prefix+1)
        actions = torch.tensor(actions) # f * 9(prefix+1)

        first_frame = (video[:,0].permute(1,2,0) * 0.5 + 0.5)*255 # h w 3 

        return {
            "pixel_values": video,
            "game_actions": actions,
            "first_frame": first_frame,
            "name": name
        }

    
    def __getitem__(self, index):
        h5_path = os.path.join(self.root,self.subs[index])
        try : 
            data = self.read_h5(h5_path)
            return data
        except Exception as e:
            print(f"Error reading {h5_path}: {e}")
            return self.__getitem__((index + 1) % len(self.subs))
        
            
    def __len__(self):
        return len(self.subs)


class DoomLatentDataset(Dataset):
    def __init__(self,root=None):
        super().__init__()
        self.root = '/root/leinyu/data/doom/freedoom_random_128_u2_rs_fix_v6_1280_720_hdf5'
        self.subs = os.listdir(self.root)
    
    def __getitem__(self, index):
        sub = self.subs[index]
        sub_path = os.path.join(self.root,sub)
        data  = torch.load(sub_path,map_location='cpu',weights_only=True)
        return data 

    def __len__(self):
        return len(self.subs)