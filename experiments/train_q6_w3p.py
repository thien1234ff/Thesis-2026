import os
import sys
import argparse
import tempfile
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import iresnet18, iresnet50, mobilefacenet
from src.quantizer import quantize_model
from src.core_algorithm import QATTrainingEngine

class SyntheticFaceDataset(Dataset):
    """Generates a small dummy dataset of random images for testing training execution."""
    def __init__(self, num_classes=5, num_images_per_class=10, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        for c in range(num_classes):
            for _ in range(num_images_per_class):
                # Generate random RGB image
                img = Image.fromarray((torch.rand(3, 112, 112).permute(1, 2, 0).numpy() * 255).astype('uint8'))
                self.data.append(img)
                self.labels.append(c)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    parser = argparse.ArgumentParser(description="Train 6-bit face recognition model using 3-phase QAT.")
    parser.add_argument("--arch", type=str, default="iresnet18", choices=["iresnet18", "iresnet50", "mobilefacenet"],
                        help="Model architecture name.")
    parser.add_argument("--data-dir", type=str, default="data/final_dataset",
                        help="Path to training image directory (ImageFolder format).")
    parser.add_argument("--teacher-weights", type=str, default="weights/FP32/iresnet18_fp32.pth",
                        help="Path to full precision teacher weights.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate.")
    parser.add_argument("--beta", type=float, default=300.0, help="Beta-MSE alignment weight coefficient.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--save-dir", type=str, default="weights/w-3p", help="Directory to save checkpoints.")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch with dummy data for verification.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Setup data transformation
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Check if dataset path exists
    use_synthetic = args.dry_run or not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0
    
    if use_synthetic:
        print("💡 [INFO] Real training image dataset folder not found or empty.")
        print("👉 Running in DRY-RUN mode using dynamically generated synthetic face dataset.")
        trainset = SyntheticFaceDataset(num_classes=5, num_images_per_class=20, transform=transform)
        train_loader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True)
        epochs = 2
        print(f"Dataset: Synthetic (Classes: 5, Samples: 100), running for {epochs} test epochs.")
    else:
        print(f"Loading real dataset from: {args.data_dir}")
        trainset = datasets.ImageFolder(args.data_dir, transform=transform)
        train_loader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4 if os.name != 'nt' else 0, # avoid multiprocessing issues in Windows tests
            pin_memory=True,
            drop_last=True
        )
        epochs = args.epochs
        print(f"Dataset: Real (Classes: {len(trainset.classes)}, Samples: {len(trainset)}), running for {epochs} epochs.")

    # 1. Initialize Teacher Model
    num_features = 512 if args.arch != "mobilefacenet" else 128
    if args.arch == "iresnet18":
        teacher = iresnet18(num_features=num_features)
    elif args.arch == "iresnet50":
        teacher = iresnet50(num_features=num_features)
    else:
        teacher = mobilefacenet(embedding_size=num_features)
        
    # Load teacher weights if available, otherwise initialize randomly
    if os.path.exists(args.teacher_weights):
        print(f"Loading teacher weights from: {args.teacher_weights}")
        teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device), strict=False)
    else:
        print("⚠️ Teacher weights not found. Initializing teacher randomly for structure checkout.")
        
    # 2. Initialize Student Model (Quantized)
    if args.arch == "iresnet18":
        student_fp32 = iresnet18(num_features=num_features)
    elif args.arch == "iresnet50":
        student_fp32 = iresnet50(num_features=num_features)
    else:
        student_fp32 = mobilefacenet(embedding_size=num_features)
        
    if os.path.exists(args.teacher_weights):
        student_fp32.load_state_dict(torch.load(args.teacher_weights, map_location=device), strict=False)
        
    student = quantize_model(student_fp32, weight_bit=6, act_bit=6)
    
    # 3. Setup optimizer
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # 4. Run Training Engine
    engine = QATTrainingEngine(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        optimizer=optimizer,
        base_lr=args.lr,
        beta=args.beta,
        epochs=epochs,
        device=device
    )
    
    engine.train(save_dir=args.save_dir, model_prefix=args.arch)
    print("🎉 Training finished successfully!")

if __name__ == "__main__":
    main()
