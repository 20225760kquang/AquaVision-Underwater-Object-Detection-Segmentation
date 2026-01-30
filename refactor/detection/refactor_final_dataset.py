"""
Chuẩn hóa lại các nhãn trong dataset final theo quy ước:
- Prefix 3 (0-5): crab, fish, jellyfish, shrimp, small_fish, starfish
- Prefix 4 (6-8): sea_cucumber, sea_urchin, scallop (hiện tại đang là 0, 1, 2)
- Prefix 2, 0 (9): diver (hiện tại đang là 0)
- Prefix 1 (10): string (hiện tại đang là 3)
"""
from pathlib import Path
from typing import Dict

LABELS_DIR = Path("../../dataset/detection/final/labels")
SPLITS = ["train", "val", "test"]

# Mapping từ (prefix, old_class_id) -> new_class_id
CLASS_MAPPING: Dict[str, Dict[int, int]] = {
    "0_": {0: 9},  # diver: 0 -> 9
    "1_": {3: 10},  # string: 3 -> 10
    "2_": {0: 9},  # diver: 0 -> 9
    "3_": {},  # crab, fish, jellyfish, shrimp, small_fish, starfish: giữ nguyên 0-5
    "4_": {  # sea_cucumber, sea_urchin, scallop: 0,1,2 -> 6,7,8
        0: 6,
        1: 7,
        2: 8
    }
}


def get_prefix(filename: str) -> str:
    """Lấy prefix từ tên file (ví dụ: '3_0_000042.txt' -> '3_')"""
    if "_" in filename:
        return filename.split("_")[0] + "_"
    return ""


def convert_label_line(line: str, old_to_new: Dict[int, int]) -> str:
    """Chuyển đổi class_id trong 1 dòng label YOLO"""
    parts = line.strip().split()
    if not parts:
        return line

    old_class_id = int(parts[0])
    new_class_id = old_to_new.get(old_class_id, old_class_id)  # Nếu không có trong mapping thì giữ nguyên

    parts[0] = str(new_class_id)
    return " ".join(parts) + "\n"


def process_label_file(label_path: Path) -> int:
    """Xử lý 1 file label, trả về số dòng đã thay đổi"""
    prefix = get_prefix(label_path.name)

    if prefix not in CLASS_MAPPING:
        return 0

    old_to_new = CLASS_MAPPING[prefix]
    if not old_to_new:  # Prefix 3 không cần chuyển đổi
        return 0

    # Đọc nội dung cũ
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Chuyển đổi
    changed_count = 0
    new_lines = []
    for line in lines:
        new_line = convert_label_line(line, old_to_new)
        new_lines.append(new_line)
        if new_line != line:
            changed_count += 1

    # Ghi lại nếu có thay đổi
    if changed_count > 0:
        with open(label_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return changed_count


def main():
    print("🚀 Bắt đầu chuẩn hóa nhãn...")

    total_files = 0
    total_lines_changed = 0
    stats_by_prefix = {prefix: {"files": 0, "lines": 0} for prefix in CLASS_MAPPING.keys()}

    # Duyệt qua tất cả split
    for split in SPLITS:
        split_dir = LABELS_DIR / split
        if not split_dir.exists():
            print(f"⚠️  Thư mục {split} không tồn tại, bỏ qua")
            continue

        print(f"\n📂 Đang xử lý split: {split}")

        # Duyệt qua tất cả file .txt
        for label_file in split_dir.glob("*.txt"):
            changed = process_label_file(label_file)

            if changed > 0:
                total_files += 1
                total_lines_changed += changed

                prefix = get_prefix(label_file.name)
                stats_by_prefix[prefix]["files"] += 1
                stats_by_prefix[prefix]["lines"] += changed

                print(f"  ✓ {label_file.name}: {changed} dòng đã sửa")

    # Thống kê tổng hợp
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ TỔNG HỢP")
    print("=" * 60)
    print(f"Tổng số file đã sửa: {total_files}")
    print(f"Tổng số dòng đã thay đổi: {total_lines_changed}")

    print("\n📋 Chi tiết theo prefix:")
    for prefix, stat in stats_by_prefix.items():
        if stat["files"] > 0:
            mapping_info = CLASS_MAPPING[prefix]
            if mapping_info:
                print(f"  {prefix}: {stat['files']} files, {stat['lines']} dòng | Mapping: {mapping_info}")

    print("\n✅ Hoàn thành!")


if __name__ == "__main__":
    main()
