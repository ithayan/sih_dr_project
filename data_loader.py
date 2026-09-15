import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==========================================
# 1. Disease Grading Dataset (Classification)
# ==========================================
class IDRiD_Grading_Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir: Path to 'B. Disease Grading' folder
        split: 'train' or 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Paths based on standard IDRiD folder structure
        if split == 'train':
            self.img_dir = os.path.join(root_dir, '1. Original Images', 'a. Training Set')
            csv_path = os.path.join(root_dir, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
        else:
            self.img_dir = os.path.join(root_dir, '1. Original Images', 'b. Testing Set')
            csv_path = os.path.join(root_dir, '2. Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv')
            
        # Load labels
        self.labels_df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.labels_df)
        
    def __getitem__(self, idx):
        # The CSV has a column 'Image name' like 'IDRiD_001'
        img_name = self.labels_df.iloc[idx]['Image name'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        # 'Retinopathy grade' column has values 0, 1, 2, 3, 4
        grade = int(self.labels_df.iloc[idx]['Retinopathy grade'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(grade, dtype=torch.long)


# ==========================================
# 2. Localization Dataset (Regression)
# ==========================================
class IDRiD_Localization_Dataset(Dataset):
    def __init__(self, root_dir, target='optic_disc', split='train', transform=None):
        """
        root_dir: Path to 'C. Localization' folder
        target: 'optic_disc' or 'fovea'
        split: 'train' or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        
        if split == 'train':
            self.img_dir = os.path.join(root_dir, '1. Original Images', 'a. Training Set')
            prefix = 'Training'
        else:
            self.img_dir = os.path.join(root_dir, '1. Original Images', 'b. Testing Set')
            prefix = 'Testing'
            
        if target == 'optic_disc':
            csv_name = f'IDRiD_Optic_Disc_Center_{prefix}_Labels.csv'
        else:
            csv_name = f'IDRiD_Fovea_Center_{prefix}_Labels.csv'
            
        csv_path = os.path.join(root_dir, '2. Groundtruths', csv_name)
        self.labels_df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['Image name'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        x_coord = float(self.labels_df.iloc[idx]['X_Coordinate'])
        y_coord = float(self.labels_df.iloc[idx]['Y_Coordinate'])
        coords = torch.tensor([x_coord, y_coord], dtype=torch.float32)
        
        # Note: If resizing images, coordinates MUST be scaled proportionally.
        # Keeping transforms simple here, but custom coordinate scaling is needed if resizing.
        if self.transform:
            image = self.transform(image)
            
        return image, coords


# ==========================================
# 3. Blood Vessel Segmentation Dataset (DRIVE)
# ==========================================
class DRIVE_Segmentation_Dataset(Dataset):
    def __init__(self, root_dir, split='training', transform=None, mask_transform=None):
        """
        root_dir: Path to 'datasets' folder (which contains 'training' and 'test' folders)
        split: 'training' or 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Handle the nested structure (e.g., datasets/training/training/images)
        base_path = os.path.join(root_dir, split, split)
        self.img_dir = os.path.join(base_path, 'images')
        self.manual_dir = os.path.join(base_path, '1st_manual')
        self.fov_mask_dir = os.path.join(base_path, 'mask')
        
        if os.path.exists(self.img_dir):
            self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.tif')])
        else:
            self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        base_name = img_name.split('_')[0]
        
        # Match standard DRIVE naming conventions
        manual_name = f"{base_name}_manual1.gif"
        if self.split == 'training':
            fov_name = f"{base_name}_training_mask.gif"
        else:
            fov_name = f"{base_name}_test_mask.gif"
            
        manual_path = os.path.join(self.manual_dir, manual_name)
        
        image = Image.open(img_path).convert("RGB")
        manual_mask = Image.open(manual_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            manual_mask = self.mask_transform(manual_mask)
            
        return image, manual_mask

# ==========================================
# Example Execution Block
# ==========================================
if __name__ == "__main__":
    # Define standard ImageNet transforms for lightweight CNNs (MobileNetV2, etc.)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Test Disease Grading Loader
    grading_path = r"C:\Users\91900\Downloads\B. Disease Grading"
    if os.path.exists(grading_path):
        try:
            grading_dataset = IDRiD_Grading_Dataset(root_dir=grading_path, split='train', transform=base_transform)
            print(f"✅ Loaded Disease Grading Dataset: {len(grading_dataset)} training images.")
            
            # Fetch a sample
            img, label = grading_dataset[0]
            print(f"   -> Sample 0 Shape: {img.shape}, DR Grade: {label.item()} (Class {label.item()})")
        except Exception as e:
            print(f"❌ Error loading Disease Grading: {e}")
    else:
        print(f"⚠️ Path not found: {grading_path}")

    print("-" * 50)

    # 2. Test Localization Loader
    loc_path = r"C:\Users\91900\Downloads\C. Localization"
    if os.path.exists(loc_path):
        try:
            # Note: In a real training script, you must write a custom transform to scale the X/Y coordinates
            # if you resize the image to 224x224. This loader provides the raw coordinates from the CSV.
            loc_dataset = IDRiD_Localization_Dataset(root_dir=loc_path, target='optic_disc', split='train', transform=transforms.ToTensor())
            print(f"✅ Loaded Localization Dataset (Optic Disc): {len(loc_dataset)} training images.")
            
            img, coords = loc_dataset[0]
            print(f"   -> Sample 0 Shape: {img.shape}, Coordinates: {coords.tolist()}")
        except Exception as e:
            print(f"❌ Error loading Localization: {e}")
    else:
        print(f"⚠️ Path not found: {loc_path}")

    print("-" * 50)

    # 3. Test DRIVE Segmentation Loader
    drive_path = r"C:\Users\91900\Downloads\datasets"
    if os.path.exists(drive_path):
        try:
            # We need a different transform for the mask (just resize and tensor, no normalization)
            mask_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
            
            drive_dataset = DRIVE_Segmentation_Dataset(root_dir=drive_path, split='training', transform=base_transform, mask_transform=mask_transform)
            print(f"✅ Loaded DRIVE Segmentation Dataset: {len(drive_dataset)} training images.")
            
            if len(drive_dataset) > 0:
                img, mask = drive_dataset[0]
                print(f"   -> Sample 0 Image Shape: {img.shape}, Mask Shape: {mask.shape}")
        except Exception as e:
            print(f"❌ Error loading DRIVE dataset: {e}")
    else:
        print(f"⚠️ Path not found: {drive_path}")
