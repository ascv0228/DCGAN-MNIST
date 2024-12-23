import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torchinfo import summary
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from q2_train.hyperparameters import *
from q2_train.custom_dataset import CustomDataset

if ngf == 64:
    from q2_train.network_64 import Generator, Discriminator
else:
    from q2_train.network_28 import Generator, Discriminator

def create_image_grid(images, grid_size=(8, 8), image_size=(28, 28)):
    rows, cols = grid_size
    img_h, img_w = image_size

    # 初始化空白的網格畫布（height, width, 3）以容納 RGB 圖像
    grid_image = np.zeros((rows * img_h, cols * img_w))
    
    for idx, image in enumerate(images):
        row = idx // cols  # 計算當前圖像的行位置
        col = idx % cols   # 計算當前圖像的列位置
        grid_image[row * img_h : (row + 1) * img_h, col * img_w : (col + 1) * img_w] = image
    
    return grid_image

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow")
        self.resize(600, 400)

        # 左邊按鈕
        self.show_training_button = QPushButton("1. Show Training Images")
        self.show_model_button = QPushButton("2. Show Model Structure")
        self.show_loss_button = QPushButton("3. Show Training Loss")
        self.inference_button = QPushButton("4. Inference")

        self.show_training_button.clicked.connect(self.show_training)
        self.show_model_button.clicked.connect(self.show_model)
        self.show_loss_button.clicked.connect(self.show_loss)
        self.inference_button.clicked.connect(self.inference)

        # 左邊垂直佈局
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.show_training_button)
        left_layout.addWidget(self.show_model_button)
        left_layout.addWidget(self.show_loss_button)
        left_layout.addWidget(self.inference_button)

        # 主佈局
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)

        self.setLayout(main_layout)

        cudnn.benchmark = True

        #set manual seed to a constant get a consistent output
        manualSeed = random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        self.D = Discriminator(ngpu=1).eval()
        self.G = Generator(ngpu=1).eval()

        base_folder = f"./q2_train/weights_{ngf}"
        # load weights
        self.D.load_state_dict(torch.load(f'{base_folder}/netD_epoch_29.pth'))
        self.G.load_state_dict(torch.load(f'{base_folder}/netG_epoch_29.pth'))
        if torch.cuda.is_available():
            self.D = self.D.cuda()
            self.G = self.G.cuda()

        self.batch_size = 64

    def show_training(self):
        image_folder = "../cvdl_hw2_data/Q2_images/data/mnist"
        if not os.path.exists(image_folder):
            print("資料夾不存在:", image_folder)
            return
        
        image_list = os.listdir(image_folder)
        random_image_list = random.sample(image_list, self.batch_size)
        original_images = list(map(lambda x: Image.open(os.path.join(image_folder, x)).convert('L'), random_image_list))

        transform = transforms.Compose([
            transforms.RandomRotation(60),
            transforms.ToTensor(),  # 轉換成 Tensor
            transforms.ToPILImage()  # 再轉回 PIL.Image 以便顯示
        ])

        augmented_images = list(map(transform, original_images))

        original_image = create_image_grid(original_images)
        augmented_image = create_image_grid(augmented_images)

        # 創建一個 1x2 的子圖
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image, cmap='gray')
        axes[0].axis('off')
        axes[1].imshow(augmented_image, cmap='gray')
        axes[1].axis('off')

        # 設定子圖標題
        axes[0].set_title("Training Dataset (Original)")
        axes[1].set_title("Training Dataset (Augmented)")

        plt.tight_layout()
        plt.show()

    def show_model(self):
        print(self.G) 
        print(self.D) 

    def show_loss(self):
        loss_image_path = "./q2_result/G_D_loss.png"
        img = Image.open(loss_image_path)
        plt.figure()
        plt.imshow(img)
        plt.xticks([]) 
        plt.yticks([]) 
        plt.show()

    def inference(self):
        
        image_folder = "../cvdl_hw2_data/Q2_images/data/mnist"
        if not os.path.exists(image_folder):
            print("資料夾不存在:", image_folder)
            return
        
        image_list = os.listdir(image_folder)
        random_image_list = random.sample(image_list, self.batch_size)
        original_images = list(map(lambda x: Image.open(os.path.join(image_folder, x)).convert('L'), random_image_list))

        latent_size = 100
        fixed_noise = torch.randn(self.batch_size, latent_size, 1, 1)

        if torch.cuda.is_available():
            fixed_noise = fixed_noise.cuda()

        fake_images = self.G(fixed_noise).cpu().detach().numpy()
        fake_images.squeeze(axis=1)

        print("fake image size", fake_images.shape)

        # augmented_images = list(map(transform, original_images))

        original_image = create_image_grid(original_images)
        fake_image = create_image_grid(fake_images, image_size=(ngf, ngf))

        # 創建一個 1x2 的子圖
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image, cmap='gray')
        axes[0].axis('off')
        axes[1].imshow(fake_image, cmap='gray')
        axes[1].axis('off')

        # 設定子圖標題
        axes[0].set_title("Real images")
        axes[1].set_title("Fake images")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())