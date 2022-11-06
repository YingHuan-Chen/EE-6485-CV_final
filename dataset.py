import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import imageio
from PIL import Image
import numpy as np
import os
import json
from einops import rearrange,repeat

class YoutubeHighlightDataset(Dataset):
    def __init__(self, dataset = 'YouTube_Highlights', split='train', category = 'surfing', data_path = 'data', L = 32):
        self.dataset = dataset
        self.split = split
        self.category = category
        self.video_path = os.path.join(data_path,self.split,self.category)
        self.L = L

        self.H = 224
        self.W = 384

        self.segment_name = list()
        self.video_names = list()
        #self.segment2index = dict()
        self.labels = list()
        #self.frames = list()
        self.indexs = list()
        self.segment2frame = dict()

        self.to_tensor = transforms.ToTensor()

        #counter = 0
        for video_id, video_name in enumerate(sorted(os.listdir(self.video_path))):
            try:
                label_file = open(os.path.join(self.video_path, video_name, 'match_label.json'), 'r')
            except:
                continue
            video_label = json.load(label_file)[1]
            tmp_video_listing = os.listdir(os.path.join(self.video_path, video_name))
            tmp_video_listing.remove('match_label.json')

            for segment_id, segment_name in enumerate(sorted(tmp_video_listing)):
                segment_path = os.path.join(self.video_path, video_name, segment_name)
                frames_list = list()
                for frame_id, frame_name in enumerate(sorted(os.listdir(segment_path))):
                    frames_list.append(frame_name)
                    self.indexs.append(frame_id)
                    self.segment_name.append(segment_path)
                    self.video_names.append(os.path.join(self.video_path, video_name))
                    #self.segment2index[segment_path] = video_id
                    self.labels.append((video_label[segment_id] if video_label[segment_id]!= -1  else 0))
                    #counter = counter + 1
                    #print(counter)
                
                self.segment2frame[segment_path] = frames_list


        self.labels = np.array(self.labels, dtype=int)

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx):
        current_frame_list = self.segment2frame[self.segment_name[idx]]
        frames = list()
        if self.indexs[idx] < self.L-1:
            num_pad = self.L-1-self.indexs[idx]
            pad = torch.zeros(num_pad, 3, self.H, self.W)
            L_frames = current_frame_list[:self.indexs[idx]+1]
            for frame in L_frames:
                img = imageio.imread(os.path.join(self.segment_name[idx],frame))
                img = Image.fromarray(img).resize((self.W, self.H))
                img = self.to_tensor(img)
                frames.append(img)

            input_frame = torch.stack(frames)
            input_frame = torch.cat((pad, input_frame), dim=0)
        else:
            L_frames = current_frame_list[self.indexs[idx]-(self.L-1):self.indexs[idx]+1]
            for frame in L_frames:
                img = imageio.imread(os.path.join(self.segment_name[idx],frame))
                img = Image.fromarray(img).resize((self.W, self.H))
                img = self.to_tensor(img)
                frames.append(img)
            input_frame = torch.stack(frames)

        input_frame = rearrange(input_frame, 'L c h w -> c L h w')

        return input_frame, self.labels[idx]

if __name__ == "__main__":
    data_path = 'data/YouTube_Highlights_processed'
    dataset = YoutubeHighlightDataset(dataset = 'YouTube_Highlights', split='train', category = 'dog', data_path = data_path)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            num_workers=4,
                            batch_size=1,
                            pin_memory=True)
    
    for data in dataloader:
        input_frame, label = data
        print(input_frame.shape)
        print(label)
        break


    # generate smap
    from TASED import TASED_v2 
    from scipy.ndimage.filters import gaussian_filter
    import cv2

    model = TASED_v2()
    file_weight = './TASED_updated.pt'

    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')
    
    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()
    smap = model(input_frame.cuda()).cpu().data[0]
    smap = (smap.numpy()*255.).astype(np.int)/255.
    smap = gaussian_filter(smap, sigma=7)
    print(smap.shape)
    save_img = 255*input_frame[0][:,0,:,:].cpu().numpy()
    save_img = rearrange(save_img , 'c h w -> h w c')
    print(save_img.shape)
    cv2.imwrite('smap.png', (smap/np.max(smap)*255.).astype(np.uint8))
    cv2.imwrite('image.png',(save_img.astype(np.uint8)))

