import os
import re


def handle_filename(file_name):
    split_result = re.split(r'[.]', file_name)[0]
    next_split = split_result.split("_jpg")[0].split("_")
    date_part = "".join(next_split[:-1])
    date_part = re.sub(r'[^0-9]', '', date_part)
    frame_part = next_split[-1]
    match = re.search(r'\d+$', frame_part)
    if not match:
        raise ValueError(f"Không tìm thấy số trong frame_part: {frame_part}")
    frame_num = match.group().zfill(6)
    return [date_part, frame_num]

def refactor_file_name(parent_folder, mode: str):
    target_folder = os.path.join(parent_folder, mode)
    target_images_folder = os.path.join(target_folder, "images")
    target_labels_folder = os.path.join(target_folder, "labels")

    date_mapping = {}
    date_count = 0

    for f in os.listdir(target_images_folder):
        if not f.endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            res = handle_filename(f)
            if res[0] not in date_mapping:
                date_mapping[res[0]] = f"{date_count}"
                date_count += 1
        except Exception as e:
            print(f"Lỗi xử lý file {f}: {e}")
            continue

    for f in os.listdir(target_images_folder):
        if not f.endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            old_path = os.path.join(target_images_folder, f)
            res = handle_filename(f)
            new_filename = f"3_{date_mapping[res[0]]}_{res[1]}.jpg"
            new_path = os.path.join(target_images_folder, new_filename)

            if os.path.exists(new_path):
                print(f"Cảnh báo: {new_filename} đã tồn tại, bỏ qua")
                continue

            os.rename(old_path, new_path)
            print(f"Renamed: {f} -> {new_filename}")
        except Exception as e:
            print(f"Lỗi đổi tên {f}: {e}")

    for f in os.listdir(target_labels_folder):
        if not f.endswith('.txt'):
            continue
        try:
            old_path = os.path.join(target_labels_folder, f)
            res = handle_filename(f)
            new_filename = f"3_{date_mapping[res[0]]}_{res[1]}.txt"
            new_path = os.path.join(target_labels_folder, new_filename)

            if os.path.exists(new_path):
                print(f"Cảnh báo: {new_filename} đã tồn tại, bỏ qua")
                continue

            os.rename(old_path, new_path)
            print(f"Renamed: {f} -> {new_filename}")
        except Exception as e:
            print(f"Lỗi đổi tên {f}: {e}")

def refactor_delete_no_label(parent_folder):
    img_folder = os.path.join(parent_folder, "images")
    lbl_folder = os.path.join(parent_folder, "labels")

    modes = ["test", "train", "val"]
    total_deleted_imgs = 0
    total_deleted_labels = 0

    for mode in modes:
        img_mode_folder = os.path.join(img_folder, mode)
        lbl_mode_folder = os.path.join(lbl_folder, mode)

        # Duyệt qua các file label
        label_files = [f for f in os.listdir(lbl_mode_folder) if f.endswith('.txt')]

        for label_file in label_files:
            label_path = os.path.join(lbl_mode_folder, label_file)

            # Kiểm tra nếu file label trống
            if os.path.getsize(label_path) == 0:
                # Lấy tên ảnh tương ứng (thay .txt -> .jpg)
                img_name = os.path.splitext(label_file)[0] + '.jpg'
                img_path = os.path.join(img_mode_folder, img_name)

                # Xóa label trống
                os.remove(label_path)
                total_deleted_labels += 1
                print(f"Đã xóa label trống: {label_file}")

                # Xóa ảnh tương ứng nếu tồn tại
                if os.path.exists(img_path):
                    os.remove(img_path)
                    total_deleted_imgs += 1
                    print(f"Đã xóa ảnh: {img_name}")
                else:
                    print(f"Cảnh báo: Không tìm thấy ảnh {img_name}")

    print(f"\nTổng số label trống đã xóa: {total_deleted_labels}")
    print(f"Tổng số ảnh đã xóa: {total_deleted_imgs}")

# Xóa những frame quá gần nhau dẫn tới bị trùng
def refactor_duplicated_frame(parent_folder):
    img_folder = os.path.join(parent_folder, "images")
    lbl_folder = os.path.join(parent_folder, "labels")
    modes = ["test", "train", "val"]

    keep_interval = 4  # Giữ lại mỗi 4 frame
    total_deleted_imgs = 0
    total_deleted_labels = 0

    for mode in modes:
        img_mode_folder = os.path.join(img_folder, mode)
        lbl_mode_folder = os.path.join(lbl_folder, mode)

        # Lấy danh sách ảnh trực tiếp trong folder mode
        images = [f for f in os.listdir(img_mode_folder)
                  if os.path.isfile(os.path.join(img_mode_folder, f)) and
                  f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Nhóm ảnh theo folder_code (phần giữa: 0, 1, 2,...)
        grouped_images = {}
        for img_file in images:
            # Lấy folder_code từ tên file: 3_0_000042.jpg -> 0
            parts = img_file.split('_')
            if len(parts) >= 3:
                folder_code = parts[1]  # Lấy phần 0
                if folder_code not in grouped_images:
                    grouped_images[folder_code] = []
                grouped_images[folder_code].append(img_file)

        # Xử lý từng nhóm
        for folder_code, img_list in grouped_images.items():
            img_list.sort()  # Sắp xếp theo tên trong nhóm

            # Duyệt qua từng ảnh với counter
            for counter, img_file in enumerate(img_list):
                # Giữ lại ảnh nếu counter % 4 == 0
                if counter % keep_interval == 0:
                    continue

                # Xóa ảnh
                img_path = os.path.join(img_mode_folder, img_file)
                os.remove(img_path)
                total_deleted_imgs += 1
                print(f"Đã xóa ảnh: {img_file}")

                # Xóa label tương ứng
                img_name = os.path.splitext(img_file)[0]
                label_name = f"{img_name}.txt"
                label_path = os.path.join(lbl_mode_folder, label_name)

                if os.path.exists(label_path):
                    os.remove(label_path)
                    total_deleted_labels += 1
                    print(f"Đã xóa label: {label_name}")
                else:
                    print(f"Cảnh báo: Không tìm thấy label {label_name}")

    print(f"\nTổng số ảnh đã xóa: {total_deleted_imgs}")
    print(f"Tổng số label đã xóa: {total_deleted_labels}")

if __name__ == '__main__':
    dataset_folder = "..\\..\\dataset\\detection\\BrackishUnderwater"
    # mode = "val" # others are "test" & "train"
    # refactor_file_name(dataset_folder,mode)
    # refactor_delete_no_label(dataset_folder)
    refactor_duplicated_frame(dataset_folder)