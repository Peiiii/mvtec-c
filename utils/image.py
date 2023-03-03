import random
import numpy as np
import os,shutil,glob
from PIL import Image
import cv2
from tqdm import tqdm
from wpcv import pil_ops

def random_from_range(range_list):
    if not isinstance(range_list[0],(set,list,tuple)):
        return random_from_range([range_list])
    pairs=[]
    total_length=0
    def get_length(rg):
        return rg[1]-rg[0]
    for rg in range_list:
        length=get_length(rg)
        total_length+=length
        pairs.append([length,rg])
    start=0
    for i in range(len(pairs)):
        pairs[i][0]=[start,start+pairs[i][0]/total_length]
        start=pairs[i][0][1]
    def match(n,map_data):
        for [st,ed],rg in map_data:
            if n>=st and n<ed:
                return rg[0]+random.random()*(rg[1]-rg[0])
    r=random.random()
    return match(r,pairs)


    
    
        
class ImageTransformer:
    def __init__(self) -> None:
        self.level=1
        self.config_bank={}
        for level in [1,2,3,4,5]:
            r1=(level-1)*3
            r2=level*3
            t1=(level-1)*0.02
            t2=level*0.02
            self.config_bank[level]={}
            self.config_bank[level]['r_range']=[[r1,r2],[-r2,-r1]]
            self.config_bank[level]['t_range']=[[t1,t2],[-t2,-t1]]
    def set_level(self,level):
        self.level=level
        
    def transform(self,img):
        w, h = img.size
        r_range=self.config_bank[self.level]['r_range']
        t_range=self.config_bank[self.level]['t_range']
        degree = random_from_range(r_range)
        offset_x=random_from_range(t_range)
        offset_y=random_from_range(t_range)
        offsets = [int(offset_x*w),int(offset_y*h)]
        x = pil_ops.rotate(img, degree, expand=False)
        x = pil_ops.translate(x, offsets,fillcolor=0)
        return x
    def transform_pair(self,*imgs):
        w, h = imgs[0].size
        r_range=self.config_bank[self.level]['r_range']
        t_range=self.config_bank[self.level]['t_range']
        degree = random_from_range(r_range)
        offset_x=random_from_range(t_range)
        offset_y=random_from_range(t_range)
        offsets = [int(offset_x*w),int(offset_y*h)]
        outs=[]
        for img in imgs:
            x = pil_ops.rotate(img, degree, expand=False)
            x = pil_ops.translate(x, offsets,fillcolor=0)
            outs.append(x)
        return tuple(outs)

it=ImageTransformer()
class Dataset:
    def __init__(self,root):
        self.root=root
        self.train_path=os.path.join(self.root,'train')
        self.test_path=os.path.join(self.root,'test')
        self.gt_path=os.path.join(self.root,'ground_truth')
    def get_gt_path(self,f):
        relpath=os.path.relpath(f,self.test_path).rsplit('.',maxsplit=1)[0]+"_mask.png"
        return os.path.join(self.gt_path,relpath)
    def relpath(self,f):
        return os.path.relpath(f,self.root)
    def is_test_file_good(self,f):
        relpath = os.path.relpath(f, self.test_path)
        return relpath.strip('/').strip('\\').startswith('good')
    def load_image_files(self,relpath=""):
        return glob.glob(os.path.join(self.root,relpath)+'/**/*.png',recursive=True)
    def save_image(self,img,rel_path):
        path=os.path.join(self.root,rel_path)
        dir=os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        img.save(path)



class DatasetConverter:
    def convert_train(self,src_dataset:Dataset,dst_dataset:Dataset):
        fs=src_dataset.load_image_files("train")
        for f in tqdm(fs):
            img=it.transform(Image.open(f))
            dst_dataset.save_image(img,os.path.relpath(f,src_dataset.root))
    def convert_test(self,src_dataset:Dataset,dst_dataset:Dataset):
        fs=src_dataset.load_image_files("test")
        for f in tqdm(fs):
            if src_dataset.is_test_file_good(f):
                img=it.transform(Image.open(f))
                dst_dataset.save_image(img,src_dataset.relpath(f))
            else:
                gt_path=src_dataset.get_gt_path(f)
                img,gt=it.transform_pair(Image.open(f),Image.open(gt_path))
                dst_dataset.save_image(img,src_dataset.relpath(f))
                dst_dataset.save_image(gt,src_dataset.relpath(gt_path))








def demo():
    src=r"E:\datasets\mvtec_anomaly_detection"
    dst="E:/datasets/mvtec_c/rt"
    for cls in os.listdir(src):
        src_dataset = Dataset(os.path.join(src,cls))
        dst_dataset = Dataset(os.path.join(dst,cls))
        converter=DatasetConverter()
        converter.convert_test(src_dataset,dst_dataset)
        converter.convert_train(src_dataset,dst_dataset)
    pass
if __name__ == '__main__':
    demo()