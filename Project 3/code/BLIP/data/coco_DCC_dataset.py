import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

class coco_DCC_train(Dataset):
    def __init__(self, transform, image_root, ann_root_DCC, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root_DCC (string): directory to store the annotation file (DCC)
        '''        
        filename = 'captions_no_caption_rm_eightCluster_train2014.json'
        
        assert os.path.exists(os.path.join(ann_root_DCC,filename)), 'annotation未找到'

        self.annotation = json.load(open(os.path.join(ann_root_DCC,filename),'r'))['annotations']
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_name='train2014/COCO_train2014_'+str(ann['image_id']).zfill(12)+'.jpg'

        image_path = os.path.join(self.image_root,image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
    
class coco_DCC_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root_DCC, split,vf='',tf=''):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val':'captions_split_set_bottle_val_val_novel2014.json','test':'captions_split_set_bottle_val_test_novel2014.json'}
        if vf!='': filenames['val']=vf
        if tf!='': filenames['test']=tf
        self.split=split
        
        self.annotation = json.load(open(os.path.join(ann_root_DCC,filenames[split]),'r'))['annotations']
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    

        ann = self.annotation[index]
        
        image_name='val2014/COCO_val2014_'+str(ann['image_id']).zfill(12)+'.jpg'

        image_path = os.path.join(self.image_root,image_name)    
             
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image_id']
        
        return image, int(img_id)   
    
    