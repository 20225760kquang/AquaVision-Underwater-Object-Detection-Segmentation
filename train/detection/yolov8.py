import torch
from ultralytics import YOLO

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    print(f"Using {device}")
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=device,
        lr0=learning_rate,
        workers = 2
    )
if __name__ == "__main__":
    train()