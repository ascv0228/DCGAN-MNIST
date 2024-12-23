import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        root_dir: 圖片所在的資料夾
        transform: 用於對圖片進行的轉換
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) \
                            if fname.endswith('.png') and fname.startswith("train" if self.train else "test")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 取得檔案路徑
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 轉換為RGB格式

        # 解析檔名 (例如: test_0_61218.png)
        filename = os.path.basename(img_path)
        file_parts = filename.split('_')
        
        if len(file_parts) >= 3:  # 檢查檔名是否符合格式
            img_type = file_parts[0]  # 取得 "type" (如: 'test')
            label = int(file_parts[1])  # 取得 "label" (如: 0)
        else:
            raise ValueError(f"Filename '{filename}' does not match the required pattern <type>_<label>_<index>.png")

        # 應用圖像轉換
        if self.transform is not None:
            image = self.transform(image)

        # 返回圖像、標籤和type (如果不需要type可忽略)
        return image, label
        # return image, label, img_type

