import os
import re
import shutil

def hanlde_file_name(f):
    cut_end_file = re.split(r'[.]', f)[0]
    next = re.split(r'[_]', cut_end_file)
    sub_folder_name = '_'.join(next[:4])
    image_idx = next[-1]
    return [sub_folder_name,image_idx]

def insight_folder(parent_folder):
    img_folder = os.path.join(parent_folder, "images")
    lbl_folder = os.path.join(parent_folder, "labels")

    total_imgs = 0
    for folder in os.listdir(img_folder) :
        folder_path = os.path.join(img_folder,folder)
        files_only = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        print(f"Số lượng file có trong {folder}: {len(files_only)}")
        total_imgs += len(files_only)
    print(total_imgs)
    modes = ["test", "train", "val"]
    sub_folder = {}
    for mode in modes:
        label_folder = os.path.join(lbl_folder,mode)
        files = os.listdir(label_folder)
        for f in files :
            cut_end_file = re.split(r'[.]', f)[0]
            next = re.split(r'[_]',cut_end_file)
            sub_folder_name = '_'.join(next[:4])
            # Xóa file thừa
            first_char = sub_folder_name[:1]
            if first_char == 'p' :
                file_path = os.path.join(label_folder, f)
                os.remove(file_path)
                print(f"Đã xóa file: {f}")
            elif first_char == 'b':
                if sub_folder_name not in sub_folder:
                    sub_folder[sub_folder_name] = 1
                else :
                    sub_folder[sub_folder_name] += 1

    for name, num_files in sub_folder.items():
        print(f"Số lượng file có trong {name}: {num_files}")
# Xóa những ảnh mà file label không có nội dung
def refactor_delete_no_label(parent_folder):
    img_folder = os.path.join(parent_folder, "images")
    lbl_folder = os.path.join(parent_folder, "labels")

    modes = ["test", "train", "val"]
    target_delete_files = []
    for mode in modes:
        label_folder = os.path.join(lbl_folder, mode)
        files = os.listdir(label_folder)
        for f in files:
            file_path = os.path.join(label_folder,f)

            cut_end_file = re.split(r'[.]', f)[0]
            next = re.split(r'[_]', cut_end_file)
            sub_folder_name = '_'.join(next[:4])
            image_idx = next[-1]
            if os.path.getsize(file_path) == 0 :
                path = os.path.join(img_folder,sub_folder_name,f"{image_idx}.jpg")
                target_delete_files.append(path)
                os.remove(file_path)
    for file_to_delete in target_delete_files:
        try:
            os.remove(file_to_delete)
            print(f"Đã xóa: {file_to_delete}")
        except FileNotFoundError:
            print(f"File không tồn tại: {file_to_delete}")
        except Exception as e:
            print(f"Lỗi khi xóa {file_to_delete}: {e}")

    print(f"\nTổng số file đã xóa: {len(target_delete_files)}")


# Xóa những frame quá gần nhau dẫn tới bị trùng
def refactor_duplicated_frame(parent_folder):
    img_folder = os.path.join(parent_folder, "images")
    lbl_folder = os.path.join(parent_folder, "labels")
    modes = ["test", "train", "val"]

    keep_interval = 20  # Giữ lại mỗi 20 frame
    total_deleted_imgs = 0
    total_deleted_labels = 0

    for mode in modes:
        img_mode_folder = os.path.join(img_folder, mode)
        lbl_mode_folder = os.path.join(lbl_folder, mode)

        # Lấy danh sách ảnh trực tiếp trong folder mode
        images = [f for f in os.listdir(img_mode_folder)
                  if os.path.isfile(os.path.join(img_mode_folder, f)) and
                  f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Nhóm ảnh theo folder_code (ví dụ: 7A, 8B,...)
        grouped_images = {}
        for img_file in images:
            # Lấy folder_code từ tên file: 2_7A_000001.jpg -> 7A
            parts = img_file.split('_')
            if len(parts) >= 3:
                folder_code = parts[1]  # Lấy phần 7A
                if folder_code not in grouped_images:
                    grouped_images[folder_code] = []
                grouped_images[folder_code].append(img_file)

        # Xử lý từng nhóm
        for folder_code, img_list in grouped_images.items():
            img_list.sort()  # Sắp xếp theo tên trong nhóm

            # Duyệt qua từng ảnh với counter
            for counter, img_file in enumerate(img_list):
                # Giữ lại ảnh nếu counter % 20 == 0
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
# Đổi tên về định dạng dễ dùng
def refactor_file_name(parent_folder):
    img_folder = os.path.join(parent_folder, "images")
    lbl_folder = os.path.join(parent_folder, "labels")
    modes = ["test", "train", "val"]

    dataset_prefix = "2"  # Prefix để phân biệt dataset

    for mode in modes:
        img_mode_folder = os.path.join(img_folder, mode)
        lbl_mode_folder = os.path.join(lbl_folder, mode)

        # Duyệt qua từng subfolder
        for sub_folder in os.listdir(img_mode_folder):
            sub_folder_path = os.path.join(img_mode_folder, sub_folder)

            if not os.path.isdir(sub_folder_path):
                continue

            # Lấy số thứ tự và ký tự từ tên subfolder
            # Ví dụ: barbados_scuba_007_A -> 7A
            parts = sub_folder.split('_')
            if len(parts) >= 4:
                folder_code = parts[2].lstrip('0') + parts[3]  # "007" -> "7", + "A" = "7A"
            else:
                folder_code = sub_folder  # Fallback nếu format khác

            # Duyệt qua từng file ảnh trong subfolder
            for img_file in os.listdir(sub_folder_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                # Lấy số thứ tự từ tên file cũ
                old_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]

                # Format số thứ tự thành 6 chữ số
                frame_number = old_name.zfill(6)

                # Tạo tên mới: 2_7A_000001.jpg
                new_name = f"{dataset_prefix}_{folder_code}_{frame_number}{ext}"

                # Đường dẫn cũ và mới cho ảnh
                old_img_path = os.path.join(sub_folder_path, img_file)
                new_img_path = os.path.join(img_mode_folder, new_name)

                # Di chuyển và đổi tên ảnh
                shutil.move(old_img_path, new_img_path)
                print(f"Đã di chuyển ảnh: {img_file} -> {new_name}")

                # Đổi tên file label tương ứng
                old_label_name = f"{sub_folder}_{old_name}.txt"
                new_label_name = f"{dataset_prefix}_{folder_code}_{frame_number}.txt"

                old_label_path = os.path.join(lbl_mode_folder, old_label_name)
                new_label_path = os.path.join(lbl_mode_folder, new_label_name)

                if os.path.exists(old_label_path):
                    shutil.move(old_label_path, new_label_path)
                    print(f"Đã đổi tên label: {old_label_name} -> {new_label_name}")
                else:
                    print(f"Cảnh báo: Không tìm thấy label {old_label_name}")

            # Xóa subfolder rỗng sau khi di chuyển hết file
            try:
                os.rmdir(sub_folder_path)
                print(f"Đã xóa subfolder rỗng: {sub_folder}")
            except OSError:
                print(f"Không thể xóa subfolder: {sub_folder} (có thể còn file)")

    print("\nHoàn tất đổi tên và di chuyển file!")

if __name__ == '__main__':
    dataset_folder = "D:\\AI_ENGINEER\\Hust\\GR2_UOD\\dataset\\detection\\VDD"
    # insight_folder(dataset_folder)
    # refactor_delete_no_label(dataset_folder)
    # refactor_file_name(dataset_folder)
    refactor_duplicated_frame(dataset_folder)