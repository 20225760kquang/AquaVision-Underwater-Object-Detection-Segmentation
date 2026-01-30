import cv2


def cut_video(input_path, output_path, start_time, end_time, fps=25):
    """
    Cut video from start_time to end_time using OpenCV

    Args:
        input_path: Path to input video
        output_path: Path to output video
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Frames per second (default: 25)
    """
    # Open video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Không thể mở video!")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")
    print(f"Kích thước: {width}x{height}")
    print(f"Tổng số frame: {total_frames}")

    # Calculate frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    print(f"Frame bắt đầu: {start_frame}, Frame kết thúc: {end_frame}")

    # Set video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Set to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    print(f"Đang cắt video từ frame {start_frame} đến {end_frame}...")

    frame_count = 0
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        current_frame += 1
        frame_count += 1

        # Print progress every 25 frames (1 second)
        if frame_count % 25 == 0:
            progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
            print(f"Tiến độ: {progress:.1f}% - {frame_count}/{end_frame - start_frame} frames")

    # Release resources
    cap.release()
    out.release()
    print(f"Hoàn thành! Đã cắt {frame_count} frames.")
    print(f"Video đã được lưu tại: {output_path}")


if __name__ == "__main__":
    input_video = "../../source/train/rov.mp4"
    output_video = "../../source/train/rov_10_11.mp4"

    # Cut from minute 10 to minute 11 with 25 FPS
    cut_video(input_video, output_video, start_time=600, end_time=660, fps=25)