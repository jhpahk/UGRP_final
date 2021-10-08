from types import TracebackType
import torch
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import glob
from pycocotools.coco import COCO
from PIL import Image
from skimage.draw import polygon
from torchvision.transforms.functional import scale

DATA_ROOT = "./data/"
path_list = os.listdir(DATA_ROOT)

# Transform (Preprocessing)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     # Imagenet mean, std
])

'''
DATA: PoseTrack

Custom Dataset: PoseTrackDataset

    return: Video_Num, Image(T), Image(T+1), Annotation(T), Annotation(T+1)

'''

class PoseTrackDataset(Dataset):
    def __init__(self, data_root = DATA_ROOT, transform = transform, limit=-1):
        self.transform = transform
        self.data_root = data_root
        self.image_dir_path = os.path.join(self.data_root, "images/train")
        self.annot_path = os.path.join(self.data_root, "annotations/train")
        self.annot_list = [path + ".json" for path in os.listdir(self.image_dir_path) if path + ".json" in os.listdir(self.annot_path)]
        self.image_dir_path_with_annot = [path for path in os.listdir(self.image_dir_path) if path + ".json" in os.listdir(self.annot_path)]

        self.all_image_list = []
        self.coco_object = []

        self.pair_img_annot = []

        self.limit = limit

        self._create_all_img_list()
        self._create_pair_img_annot_per_vid()


    def _get_images_per_video(self, json_file_path):
        coco = COCO(os.path.join(self.annot_path, json_file_path))
        img_ids = coco.getImgIds()  # image ids (여러개)
        imgs = coco.loadImgs(img_ids)
        # vid_id = np.unique([img['vid_id'] for img in imgs])

        posetrack_images = []
        for img in imgs:
            if not img['is_labeled']:  # or img['vid_id'] != '000015':  # Uncomment to filter for a sequence.
                pass
            else:
                posetrack_images.append(img)
        return posetrack_images, coco


    def _create_all_img_list(self):
        count = 0
        for json_file_path in self.annot_list:
            if count == self.limit:
                break

            imgs, coco = self._get_images_per_video(json_file_path)
            self.all_image_list.append(imgs)
            self.coco_object.append(coco)

            if self.limit > 0:
                count += 1
            

    def _get_keypoints_img(self, img_data, track_id, anns):
        
        # Load Image
        img = Image.open(os.path.join(self.data_root, img_data['file_name']))
        
        img_x = img.size[0]
        x_scale = img_x / 256
        img_y = img.size[1]
        y_scale = img_y / 256

        img = self.transform(img)

        # Visualize ignore regions if present.
        if 'ignore_regions_x' in img_data.keys():
            for region_x, region_y in zip(img_data['ignore_regions_x'], img_data['ignore_regions_y']):
                rr, cc = polygon(region_y, region_x, img.shape)
                img[rr, cc, 1] = 128 + img[rr, cc, 1]/2


        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')

        if datasetType == 'instances':
            for ann in anns:
                if ann["track_id"] != track_id:
                    continue

                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    kp = np.array(ann['keypoints'])
                    
                    kp_pairs = []
                    for i in range(0, len(kp), 3):
                        if kp[i + 2] == 0:
                            kp_pairs.append(0)
                        else:
                            kp_pairs.append((int(kp[i] // x_scale), int(kp[i + 1] // y_scale)))
                            if (kp[i] // x_scale) >= 256:
                                print(img_data['file_name'])
                                print("x", img_x, x_scale, kp[i])
                            if (kp[i + 1] // y_scale) >= 256:
                                print(img_data['file_name'])
                                print("y", img_y, y_scale, kp[i + 1])

        return (img, kp_pairs)
        

    def _create_pair_img_annot_per_vid(self):
        count = 0
        for idx, vid_path in enumerate(self.image_dir_path_with_annot):
            if count == self.limit:
                break

            print("Processing {}th video".format(idx + 1))
            coco = self.coco_object[idx]
            num_valid_img = len(self.all_image_list[idx])

            first_image_data = self.all_image_list[idx][0]
            ann_ids = coco.getAnnIds(imgIds = first_image_data['id'])
            anns = coco.loadAnns(ann_ids)

            num_human = np.unique([ann['track_id'] for ann in anns])
            
            for i, frame in enumerate(self.all_image_list[idx]):
                print("-Processing {}th frame".format(i + 1))

                if i + 2 >= num_valid_img:
                    break
                image_data_0 = self.all_image_list[idx][i]
                ann_ids_0 = coco.getAnnIds(imgIds = image_data_0['id'])
                anns_0 = coco.loadAnns(ann_ids_0)

                image_data_1 = self.all_image_list[idx][i+1]
                ann_ids_1 = coco.getAnnIds(imgIds = image_data_1['id'])
                anns_1 = coco.loadAnns(ann_ids_1)

                human_0 = np.unique([ann['track_id'] for ann in anns_0])
                human_1 = np.unique([ann['track_id'] for ann in anns_1])

                for track_id in num_human:
                    if track_id not in human_0 or track_id not in human_1:
                        continue
                    self.pair_img_annot.append((self._get_keypoints_img(self.all_image_list[idx][i], track_id, anns), 
                                               self._get_keypoints_img(self.all_image_list[idx][i+1], track_id, anns)))
            
            if self.limit > 0:
                count += 1

    def __getitem__(self, idx):
        return self.pair_img_annot[idx]

    def __len__(self):
        return len(self.pair_img_annot)


if __name__ == "__main__":
    dataset = PoseTrackDataset(limit=1)
    dataset._create_all_img_list()

    dataset._create_pair_img_annot_per_vid()
    print(len(dataset.pair_img_annot))

    print("Test: {}".format(dataset[0]))
    print("Dataset Length: {}".format(len(dataset)))