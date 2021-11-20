import numpy as np
from PIL import Image
import torch.utils.data as data


class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform0=None, transform1=None, colorIndex=None, rgbirIndex=None, thermalIndex=None):
        data_dir = 'D:/monica/Datasets/SYSU-MM01/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        
        train_rgbir_image = np.load(data_dir + 'train_rgbir_resized_img.npy')
        self.train_rgbir_label = np.load(data_dir + 'train_rgbir_resized_label.npy')
        
        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_rgbir_image = train_rgbir_image
        self.train_thermal_image = train_thermal_image
        self.transform0 = transform0
        self.transform1 = transform1
        self.cIndex = colorIndex
        self.rIndex = rgbirIndex
        self.tIndex = thermalIndex
    
    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img3, target3 = self.train_rgbir_image[self.rIndex[index]], self.train_rgbir_label[self.rIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform0(img1)
        img3 = self.transform1(img3)
        img2 = self.transform0(img2)
        
        return img1, img3, img2, target1, target3, target2
    
    def __len__(self):
        return len(self.train_color_label)
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform0=None, transform1=None, colorIndex=None, rgbirIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = 'D:/monica/Datasets/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'
        
        color_img_file, train_color_label = load_data(train_color_list)
        
        rgbir_file,train_rgbir_label = load_data(train_color_list)
        rgbir_img_file = []
        for imgname in rgbir_file:
            a = imgname[7:]
            b = 'rgbir' + a
            rgbir_img_file.append(b)
        
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        # rgbir_file, train_rgbir_label = load_data(train_thermal_list)
        # rgbir_img_file = []
        # for imgname in rgbir_file:
        #     a = imgname[7:]
        #     b = 'rgbir' + a
        #     rgbir_img_file.append(b)
        
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)
        
        train_rgbir_image = []
        for i in range(len(rgbir_img_file)):
            img = Image.open(data_dir + rgbir_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_rgbir_image.append(pix_array)
        train_rgbir_image = np.array(train_rgbir_image)
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label
        
        self.train_rgbir_image = train_rgbir_image
        self.train_rgbir_label = train_rgbir_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform0 = transform0
        self.transform1 = transform1
        self.cIndex = colorIndex
        self.cIndex = rgbirIndex
        self.tIndex = thermalIndex
    
    def __getitem__(self, index):
        
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img3, target3 = self.train_rgbir_image[self.cIndex[index]], self.train_rgbir_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform0(img1)
        img3 = self.transform1(img3)
        img2 = self.transform0(img2)
        
        return img1, img3, img2, target1, target3, target2
    
    def __len__(self):
        return len(self.train_color_label)
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label