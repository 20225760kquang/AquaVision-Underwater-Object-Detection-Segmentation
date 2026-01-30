"""
Tổng hợp dataset từ 5 nguồn cho bài toán Object Detection
    1. Brackish Underwater : https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset
    2. UOD Dataset : https://github.com/mousecpn/Collection-of-Underwater-Object-Detection-Dataset
    3. Video Diver Dataset (VDD-C) : https://conservancy.umn.edu/items/3b926c46-c328-40d8-841f-069ef8ab1ce2
    4. Youtube - Self Annotation : https://www.youtube.com/watch?v=NDT-eAYmxY0&t=4s
    5. ROV Source - Self Annotation
"""
import shutil
from pathlib import Path

# Đường dẫn các dataset nguồn
SOURCE_DATASETS = [
    "../../dataset/detection/temps/BrackishUnderwater",
    "../../dataset/detection/temps/UOD",
    "../../dataset/detection/temps/VDD",
    "../../dataset/detection/temps/Youtube&ROV"
]

# Đường dẫn dataset đích
FINAL_DATASET_PATH = Path("../../dataset/detection/final")

# Các split có thể có
SPLITS = ["train", "val", "test"]
DATA_TYPES = ["images", "labels"]


def create_target_structure():
    """Tạo cấu trúc thư mục đích"""
    for data_type in DATA_TYPES:
        for split in SPLITS:
            target_dir = FINAL_DATASET_PATH / data_type / split
            target_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Đã tạo cấu trúc thư mục tại {FINAL_DATASET_PATH}")


def copy_files(source_dir: Path, target_dir: Path):
    """Copy tất cả file từ source sang target"""
    if not source_dir.exists():
        return 0

    count = 0
    for file in source_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, target_dir / file.name)
            count += 1
    return count


def merge_datasets():
    """Gộp tất cả dataset vào multi_source"""
    total_stats = {split: {"images": 0, "labels": 0} for split in SPLITS}

    for source_dataset in SOURCE_DATASETS:
        source_path = Path(source_dataset)
        dataset_name = source_path.name

        print(f"\n📂 Đang xử lý dataset: {dataset_name}")

        for data_type in DATA_TYPES:
            for split in SPLITS:
                source_dir = source_path / data_type / split
                target_dir = FINAL_DATASET_PATH / data_type / split

                if source_dir.exists():
                    count = copy_files(source_dir, target_dir)
                    total_stats[split][data_type] += count
                    if count > 0:
                        print(f"  ✓ {split}/{data_type}: {count} files")
                else:
                    print(f"  - {split}/{data_type}: không tồn tại (bỏ qua)")

    # In thống kê tổng hợp
    print("\n" + "=" * 50)
    print("THỐNG KÊ TỔNG HỢP")
    print("=" * 50)
    for split in SPLITS:
        img_count = total_stats[split]["images"]
        lbl_count = total_stats[split]["labels"]
        if img_count > 0 or lbl_count > 0:
            print(f"{split.upper():5s}: {img_count:5d} images, {lbl_count:5d} labels")


def main():
    print("🚀 Bắt đầu tổng hợp dataset...")

    # Tạo cấu trúc thư mục
    create_target_structure()

    # Gộp dataset
    merge_datasets()

    print("\n✅ Hoàn thành!")


if __name__ == "__main__":
    main()

