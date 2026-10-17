import os
import argparse
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO

class DroneModelTrainer:

    def __init__(self, data_yaml: str = 'data/dataset/data.yaml', model_weights: str = 'yolo26x.pt'):
        self.data_yaml = data_yaml
        self.model_weights = model_weights
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def train(self, epochs: int = 50, imgsz: int = 640, batch: int = 8, project: str = 'runs/train', name: str = 'yolo26x_drone_fire_windows', half: bool = True, workers: int = 4):
        print(f'[Trainer] Initializing YOLO26x with base weights: {self.model_weights}')
        print(f"[Trainer] Device: {(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')}")
        print(f'[Trainer] Target dataset: {self.data_yaml} | Image Size: {imgsz} | Batch Size: {batch}')
        model = YOLO(self.model_weights)
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            workers=workers,
            half=half,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            perspective=0.0005,
            degrees=10.0,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.3,
            fliplr=0.5,
            flipud=0.0,
            scale=0.5,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3.0,
            val=True,
            save=True,
            plots=True
        )
        print(f"[Trainer] Training completed! Best model saved at: {os.path.join(project, name, 'weights', 'best.pt')}")
        return results

    def validate(self, weights_path: str = 'weights/yolo26x_fire_windows.pt', imgsz: int = 640):
        model = YOLO(weights_path)
        metrics = model.val(data=self.data_yaml, imgsz=imgsz, device=self.device)
        print(f'[Trainer] Validation mAP50-95: {metrics.box.map:.4f} | mAP50: {metrics.box.map50:.4f}')
        return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CV2.0 Drone YOLO26x Model Trainer')
    parser.add_argument('--data', type=str, default='data/dataset/data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo26x.pt', help='Base model weights')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    trainer = DroneModelTrainer(args.data, args.model)
    trainer.train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
