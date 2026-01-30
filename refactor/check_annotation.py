
import os
import cv2
import json
import shutil
categories = ["Diver", "Cable", "Coral", "String", "Fish"]


def get_video_info(video_path):
    """
    Lấy thông tin chi tiết về một video file

    Args:
        video_path (str): Đường dẫn đến file video

    Returns:
        dict: Dictionary chứa thông tin video, hoặc None nếu không mở được video
    """
    # Mở video
    cap = cv2.VideoCapture(video_path)

    # Kiểm tra xem video có mở được không
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return None

    # Lấy các thông tin của video
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'duration_seconds': 0,
        'duration_formatted': '00:00:00'
    }

    # Tính thời lượng video
    if info['fps'] > 0:
        info['duration_seconds'] = info['frame_count'] / info['fps']

        # Format thời lượng thành HH:MM:SS
        hours = int(info['duration_seconds'] // 3600)
        minutes = int((info['duration_seconds'] % 3600) // 60)
        seconds = int(info['duration_seconds'] % 60)
        info['duration_formatted'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Chuyển đổi codec sang dạng string
    codec_int = info['codec']
    info['codec_string'] = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])

    # Đóng video
    cap.release()

    return info

def show_list_imgs(image_folder,bbox_folder,from_index,num_frames,result_dir):
    # Tạo folder đích
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    image_files = os.listdir(image_folder)
    bbox_files = os.listdir(bbox_folder)
    for idx in range(num_frames):
        image_file = os.path.join(image_folder, image_files[from_index+idx])
        ori_image = cv2.imread(image_file)
        height, width = ori_image.shape[:2]
        print(f"Image size {width} * {height}")
        bbox = os.path.join(bbox_folder,bbox_files[from_index+idx])
        with open(bbox, 'r') as f:
            lines = f.readlines()
            for line in lines:
                res = [float(x) for x in line.split()]
                class_index = categories[int(res[0])]
                x_center = res[1] * width
                y_center = res[2] * height
                w = res[3] * width
                h = res[4] * height
                print(f"{x_center} {y_center} {w} {h}")
                cv2.rectangle(ori_image, (int(x_center-w/2),int(y_center+h/2)), (int(x_center+w/2),int(y_center-h/2)), (0,0,255), 2)
                cv2.putText(ori_image, class_index, (int(x_center-w/2),int(y_center-h/2)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),1)
            cv2.imwrite(f"output\\checked_{from_index+idx}.jpg", ori_image)
            print("Successfully")


