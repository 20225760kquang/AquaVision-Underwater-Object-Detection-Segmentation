import os
import numpy as np
import cv2
import re
import shutil

def extract_annotated_frame(source,path,result_dir):
    # Tạo folder đích
    # if os.path.exists(result_dir):
    #     shutil.rmtree(result_dir)
    # os.makedirs(result_dir, exist_ok=True)
    # os.makedirs(os.path.join(result_dir, "images"), exist_ok=True)
    # os.makedirs(os.path.join(result_dir, "labels"), exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening video file: {source}")
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    annotated_frames = []
    labeled_file_name = []
    label_dir = os.path.join(path,"labels/Train")
    for idx,file_name in enumerate(os.listdir(label_dir)):
        annotated_frames.append(int(re.split(r'[_.]',file_name)[1]))
        labeled_file_name.append(file_name)
    frame_index = 0
    annotated_frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("UOD", frame)
        if frame_index == annotated_frames[annotated_frame_index]:
            #Chỉ lưu frame đã được gán nhãn
            file_name = f"1_{frame_index:06d}.jpg"
            save_path = os.path.join(result_dir,"images",file_name)
            cv2.imwrite(save_path, frame)
            # Copy nội dung file.txt sang thư mục đích
            src_file = os.path.join(label_dir,f"frame_{frame_index:06d}.txt")
            shutil.copy(
                src_file,
                os.path.join(result_dir,"labels", f"1_{frame_index:06d}.txt")
            )
            annotated_frame_index += 1
        frame_index +=1
        if (cv2.waitKey(1) & 0xFF == ord('q')) or frame_index > annotated_frames[-1] :
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"This video has {frame_count} frames")

def refactor_train_val_split(parent_folder):
    img_folder = os.path.join(parent_folder, "images")
    lbl_folder = os.path.join(parent_folder, "labels")

    # Tạo folder train và val
    img_train_folder = os.path.join(img_folder, "train")
    img_val_folder = os.path.join(img_folder, "val")
    lbl_train_folder = os.path.join(lbl_folder, "train")
    lbl_val_folder = os.path.join(lbl_folder, "val")

    os.makedirs(img_train_folder, exist_ok=True)
    os.makedirs(img_val_folder, exist_ok=True)
    os.makedirs(lbl_train_folder, exist_ok=True)
    os.makedirs(lbl_val_folder, exist_ok=True)

    # Lấy danh sách ảnh trực tiếp trong folder images
    images = [f for f in os.listdir(img_folder)
              if os.path.isfile(os.path.join(img_folder, f)) and
              f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Nhóm ảnh theo prefix (0_ và 1_)
    images_0 = [f for f in images if f.startswith('0_')]
    images_1 = [f for f in images if f.startswith('1_')]

    images_0.sort()
    images_1.sort()

    # Shuffle từng nhóm
    np.random.shuffle(images_0)
    np.random.shuffle(images_1)

    # Chia 80% train, 20% val cho từng nhóm
    train_count_0 = int(len(images_0) * 0.8)
    train_count_1 = int(len(images_1) * 0.8)

    train_images = images_0[:train_count_0] + images_1[:train_count_1]
    val_images = images_0[train_count_0:] + images_1[train_count_1:]

    # Di chuyển ảnh và label vào train
    for img_file in train_images:
        img_src = os.path.join(img_folder, img_file)
        img_dst = os.path.join(img_train_folder, img_file)
        shutil.move(img_src, img_dst)

        # Di chuyển label tương ứng
        img_name = os.path.splitext(img_file)[0]
        label_file = f"{img_name}.txt"
        lbl_src = os.path.join(lbl_folder, label_file)
        lbl_dst = os.path.join(lbl_train_folder, label_file)

        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst)
        else:
            print(f"Cảnh báo: Không tìm thấy label {label_file}")

    # Di chuyển ảnh và label vào val
    for img_file in val_images:
        img_src = os.path.join(img_folder, img_file)
        img_dst = os.path.join(img_val_folder, img_file)
        shutil.move(img_src, img_dst)

        # Di chuyển label tương ứng
        img_name = os.path.splitext(img_file)[0]
        label_file = f"{img_name}.txt"
        lbl_src = os.path.join(lbl_folder, label_file)
        lbl_dst = os.path.join(lbl_val_folder, label_file)

        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst)
        else:
            print(f"Cảnh báo: Không tìm thấy label {label_file}")

    print(f"Tổng số ảnh 0_: {len(images_0)}")
    print(f"  Train: {train_count_0} ảnh ({train_count_0/len(images_0)*100:.1f}%)")
    print(f"  Val: {len(images_0)-train_count_0} ảnh ({(len(images_0)-train_count_0)/len(images_0)*100:.1f}%)")
    print(f"\nTổng số ảnh 1_: {len(images_1)}")
    print(f"  Train: {train_count_1} ảnh ({train_count_1/len(images_1)*100:.1f}%)")
    print(f"  Val: {len(images_1)-train_count_1} ảnh ({(len(images_1)-train_count_1)/len(images_1)*100:.1f}%)")
    print(f"\nTổng: Train {len(train_images)} - Val {len(val_images)}")

if __name__ == '__main__':
    # source = "..\\source\\rov.mp4"
    # path = "..\\source\\quynhon"
    # result_dir = "..\\..\\dataset\\detection\\Youtube&ROV"
    # extract_annotated_frame(source,path,result_dir)
    dataset_folder = "dataset\\Detection-Task\\Youtube&ROV"
    refactor_train_val_split(dataset_folder)
