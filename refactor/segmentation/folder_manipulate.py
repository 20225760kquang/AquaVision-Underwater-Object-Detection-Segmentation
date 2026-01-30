import os
import shutil
import re

# Rename & delete sub folder
def refactor_file_name(parent_folder):
    img_folder = os.path.join(parent_folder, "new_img")
    lbl_folder = os.path.join(parent_folder, "new_mask_gt")
    modes = ["train", "val", "test"]

    dataset_prefix = "5"

    for mode in modes:
        img_mode_folder = os.path.join(img_folder, mode)
        lbl_mode_folder = os.path.join(lbl_folder, mode)

        if not os.path.isdir(img_mode_folder):
            continue

        for site in os.listdir(img_mode_folder):
            site_path = os.path.join(img_mode_folder, site)
            if not os.path.isdir(site_path):
                continue

            match = re.search(r"(\d+)", site)
            if match:
                site_code = f"s{match.group(1)}"
            else:
                site_code = "sX"  # fallback an toàn

            for img_file in os.listdir(site_path):
                if not img_file.endswith(".png"):
                    continue

                # site1_000001_016200_leftImg8bit.png
                parts = img_file.split("_")
                if len(parts) < 4:
                    # print(f"Bỏ qua file sai format: {img_file}")
                    continue

                seq = parts[1]     # 000001
                frame = parts[2]   # 016200


                new_img_name = f"{dataset_prefix}_{site_code}_{seq}_{frame}.png"
                old_img_path = os.path.join(site_path, img_file)
                new_img_path = os.path.join(img_mode_folder, new_img_name)

                shutil.move(old_img_path, new_img_path)
                # print(f"Ảnh: {img_file} -> {new_img_name}")


                old_mask_name = f"{site}_{seq}_{frame}_gtFine.png"
                old_mask_path = os.path.join(lbl_mode_folder, site, old_mask_name)

                new_mask_name = f"{dataset_prefix}_{site_code}_{seq}_{frame}_mask.png"
                new_mask_path = os.path.join(lbl_mode_folder, new_mask_name)

                if os.path.exists(old_mask_path):
                    shutil.move(old_mask_path, new_mask_path)
                    # print(f"Mask: {old_mask_name} -> {new_mask_name}")
                else:
                    continue
                    # print(f"Mask not found: {old_mask_name}")

            try:
                os.rmdir(site_path)
                os.rmdir(os.path.join(lbl_mode_folder, site))
            except OSError:
                pass

    print("\nRefactor success dataset folder")

if __name__ == '__main__':
    path = "../../dataset/segmentation/coral"
    refactor_file_name(path)