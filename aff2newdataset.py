from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from clip_transforms import *
import csv
from torchvision.transforms import Compose
from torchvision.io import write_video

class Aff2CompDatasetNew(Dataset):
    def add_video(self,info,extracted_frames_list,transform=True):
        target = str(info['vid_name'][0])
        for file in extracted_frames_list:
            if(str(file) in target):
                image_list = os.listdir(os.path.join(self.root_dir,"extracted",file))
                for i,image in enumerate(image_list):
                    if((info['vid_name'][1]) in image and "mask" not in str(file)):
                        return self.take_mask(info, file, image_list, i, image,transform)
        return self.clip_transform(np.zeros((8,112, 112, 3), dtype=np.uint8))
    def take_mask(self, info, folder, image_list, i, image,transform):
        before = i
        after = len(image_list) - i - 1
        info['start_frame'] = before
        info['end_frame'] = after
        info['path'] = os.path.join(self.root_dir,"extracted",folder,image)
        clip= np.zeros((8, 112, 112, 3), dtype=np.uint8)
        needed_before =self.video_length* self.dialation 
        last_value=self.vid_len-self.target_frame-1    
        if(before>=needed_before and after>=last_value):
            t = 0
            for z in range(i-(self.video_length-1)*self.dialation,i+1,self.dialation):
                image_path = os.path.join(self.root_dir,"extracted",folder,image_list[z])
                mask_img = Image.open(image_path)
                clip[t] = np.asarray(mask_img)
                t+=1
        return self.clip_transform(clip)

    def __init__(self,root_dir='',mtl_path = 'mtl_data/',test_set = False,target_frame = 7,vid_length = 8,dataset_dir = 'train_set.txt'):
        super(Aff2CompDatasetNew,self).__init__()
        self.bad = 0 
        self.total = 0
        self.target_frame = target_frame
        self.vid_len = vid_length
        self.video_length = vid_length
        #file lists
        self.root_dir = root_dir
        self.audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])
        self.clip_transform = Compose([RandomClipFlip(),NumpyToPyTensor()])
        self.videos =   []
        self.videos += [each for each in os.listdir(root_dir) if each.endswith(".mp4")]
        self.metadata = []
        self.metadata += [each for each in os.listdir(root_dir) if each.endswith(".json")]
        self.audio_dir = []
        self.audio_dir +=[each for each in os.listdir(root_dir) if each.endswith(".wav")]
        self.extracted_frames = []
        self.extracted_frames += [each for each in os.listdir(os.path.join(root_dir , "extracted"))]
        
        #video info
        self.clip_len = vid_length
        self.input_shape = (112,112)
        self.dialation = 6
        self.label_frame = self.clip_len * self.dialation
        csv = os.path.join(mtl_path, dataset_dir)
        self.dataset = []
        self.dataset += self.create_inputs(csv)
        self.dataset = [x for x in self.dataset if x['expressions']!=-1]
                        # self.testing = []
    def create_inputs(self,csv_path):
        labels = []
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=",")
            next(csv_reader)
            for row in csv_reader:
                labels.append(row)
        outputs = []
        for row in labels:
            vid_name = row[0].split('/')
            valience = row[1]
            arousal = row[2]
            expressions = row[3]
            action_units = row[4:]
            expected_output = {}
            expected_output['vid_name'] = vid_name
            expected_output['valience'] = float(valience)
            expected_output['arousal'] = float(arousal)
            expected_output['expressions'] = int(expressions)
            expected_output['action_units'] = [int(i) for i in action_units]
            expected_output['frame_id'] = vid_name[1]
            # expected_output['fps'] = self.get_fps(self.find_video(expected_output['vid_name']))
            outputs.append(expected_output)
        self.time_stamps = []
        
        return outputs
    def find_video(self,video_info):
        for video_name in self.videos:
            if(video_name.startswith(video_info[0])):
                return os.path.join(self.root_dir,video_name)

    def get_fps(self,video):
        video = cv2.VideoCapture(video)

        return video.get(cv2.CAP_PROP_FPS)
    
    def __getitem__(self, index):
        d = self.dataset[index]
        dict = {}
        dict['video_name'] = d['vid_name']
        dict['clip']  = self.add_video(d,self.extracted_frames)
        if(dict['clip']==None):
            # dict['clip'] = torch.ones(4,8,112,112,dtype=torch.float)
            self.bad = self.bad+1
        self.total = self.total+1
        dict['expressions'] = d['expressions']
        dict['action_units'] = d['action_units']
        dict['au0'] = dict['action_units'][0]
        dict['au1'] = dict['action_units'][1]
        dict['au2'] = dict['action_units'][2]
        dict['au3'] = dict['action_units'][3]
        dict['au4'] = dict['action_units'][4]
        dict['au5'] = dict['action_units'][5]
        dict['au6'] = dict['action_units'][6]
        dict['au7'] = dict['action_units'][7]
        dict['au8'] = dict['action_units'][8]
        dict['au9'] = dict['action_units'][9]
        dict['au10'] = dict['action_units'][10]
        dict['au11'] = dict['action_units'][11]
        dict['valience'] = d['valience']
        dict['arousal'] = d['arousal']
        dict['frame_id'] = d['frame_id']
        return dict
    def __len__(self):
        return len(self.dataset)
    def __add__(self,dict):
        self.dataset.append(dict) 
    def get_slice(self,audio,frame_offset,length):
        return audio[:,frame_offset:frame_offset+length]
  
    def __remove__(self,index):
        return self.dataset.pop(index)
if __name__ == "__main__":

    train_set= Aff2CompDatasetNew(root_dir='aff2_processed',mtl_path='mtl_data',target_frame=0)
    i = 0
    data = train_set.__getitem__(5)
    vid = data['clip'].cpu().permute(1,2,3,0)[:,:,:,:].mul(255)
    name = os.path.join(str(data['video_name'][0])+ "-" + str(data['frame_id']) + ".mp4")
    write_video(name,vid,4)
