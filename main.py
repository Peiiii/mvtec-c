import shutil
from imagecorruptions import corrupt
import cv2
import wk
import os
import glob
from tqdm import tqdm
from utils.image import Dataset, DatasetConverter, it
import os

corruption_names = [
    'gaussian_noise',
    'shot_noise',
    'defocus_blur',
    'motion_blur',
    'brightness',
    'contrast',
    'jpeg_compression',
    'geometry'
]


def make_geometry_dataset(level, src, dst):
    it.set_level(level)
    ## src: 数据集路径
    for cls in os.listdir(src):
        src_dataset = Dataset(os.path.join(src, cls))
        dst_dataset = Dataset(os.path.join(dst, cls))
        converter = DatasetConverter()
        converter.convert_test(src_dataset, dst_dataset)


def make_defect_dataset(src_path, dst_path, corruption_name, severity):
    '{cor}/{object_type}/{groud_truth}/'

    gt_files = glob.glob(src_path + '/*/ground_truth/**/*.png', recursive=True)
    good_files = glob.glob(src_path + '/*/test/good/**/*.png', recursive=True)
    test_files = glob.glob(src_path + '/*/test/**/*.png', recursive=True)

    files_to_copy = gt_files
    files_to_process = test_files
    for f in tqdm(files_to_copy):
        f2 = wk.join_path(dst_path, wk.get_relative_path(src_path, f))
        dirpath = os.path.dirname(f2)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        shutil.copy(f, f2)
    for f in tqdm(files_to_process):
        img = cv2.imread(f)
        f2 = wk.join_path(dst_path, wk.get_relative_path(src_path, f))
        cor_img = corrupt(img, severity, corruption_name)
        dirpath = os.path.dirname(f2)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        cv2.imwrite(f2, cor_img)


def make_mvtec_c(mvtec_dir, dataset_dir):
    for c in corruption_names:
        for s in range(1, 6):
            dst_path = f'{dataset_dir}/{c}_{s}'
            if os.path.exists(dst_path):
                print('Already exists: %s' % (dst_path))
                continue
            if c == "geometry":
                make_geometry_dataset(s, mvtec_dir, dst_path)
            else:
                make_defect_dataset(
                    src_path=mvtec_dir,
                    dst_path=dst_path,
                    corruption_name=c,
                    severity=s
                )


if __name__ == '__main__':
    import fire

    fire.Fire(make_mvtec_c)
