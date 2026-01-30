import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2

class UODInference:
    def __init__(self, detection_model_path, segmentation_model_path, device='cuda'):
        """
        Args:
            detection_model_path: Path to YOLOv8 detection model
            segmentation_model_path: Path to DeepLabV3 segmentation model
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load YOLOv8 detection model
        self.det_model = YOLO(detection_model_path)

        # Load DeepLabV3 segmentation model
        model = deeplabv3_resnet50(pretrained=False,aux_loss=True ,num_classes=6)
        checkpoint = torch.load(segmentation_model_path, map_location=self.device)
        self.seg_model = model
        self.seg_model.load_state_dict(checkpoint["model"])
        self.seg_model.to(self.device)
        self.seg_model.eval()

        # Segmentation transform
        self.seg_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Detection categories
        self.det_categories = {
            0: 'crab',
            1: 'fish',
            2: 'jellyfish',
            3: 'shrimp',
            4: 'small_fish',
            5: 'starfish',
            6: 'sea_cucumber',
            7: 'sea_urchin',
            8: 'scallop',
            9: 'diver',
            10: 'string'
        }
        # Segmentation categories
        self.seg_categories = {
            0: 'background',
            1: 'coral-alive',
            2: 'coral-dead',
            3: 'coral-bleached',
            4: 'algae covered substrate',
            5: 'unknown hard substrate'
        }
        # Color map for segmentation classes
        self.color_map = {
            0: (255, 255, 255),
            1: (0, 0, 255),
            2: (216, 162, 29),
            3: (0, 255, 255),
            4: (128, 178, 194),
            5: (128, 153, 161)
        }

    def detect_objects(self, image):
        """Run YOLOv8 detection"""
        results = self.det_model.predict(image, conf=0.25, iou=0.45)
        return results[0]

    def segment_image(self, image, confidence_threshold = 0.5):
        """Run DeepLabV3 segmentation"""
        # Prepare image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.seg_transform(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.seg_model(input_tensor)['out']

            # Get probabilities and max confidence
            probs = torch.softmax(output, dim=1)
            max_probs, prediction = torch.max(probs, dim=1)

            prediction[max_probs < confidence_threshold] = 0
            prediction = prediction.squeeze(0).cpu().numpy()

        # Resize back to original size
        prediction = cv2.resize(prediction, (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

        return prediction

    def draw_detection_boxes(self, image, det_results):
        """Draw bounding boxes with class names"""
        result_image = image.copy()

        # Get boxes, scores, and class IDs
        boxes = det_results.boxes.xyxy.cpu().numpy()
        scores = det_results.boxes.conf.cpu().numpy()
        class_ids = det_results.boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)

            # Get class name
            class_name = self.det_categories.get(class_id, f'Class {class_id}')

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label
            label = f'{class_name}: {score:.2f}'

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw label background
            cv2.rectangle(result_image,
                          (x1, y1 - text_height - baseline - 5),
                          (x1 + text_width, y1),
                          (0, 255, 0), -1)

            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return result_image

    def visualize_segmentation(self, image, mask):
        """Create colored segmentation overlay"""
        result = image.copy()
        overlay = np.zeros_like(image)

        for class_id, color in self.color_map.items():
            if class_id == 0:
                continue
            overlay[mask == class_id] = color

        # Blend with original image
        blend_mask = (mask != 0).astype(np.float32)
        blend_mask = np.stack([blend_mask] * 3, axis=-1)
        result = (result * (1 - blend_mask * 0.4) + overlay * (blend_mask * 0.4)).astype(np.uint8)

        # Add class labels for each segmented region
        for class_id in range(1, 6):  # Skip background (0)
            class_mask = (mask == class_id)

            if not class_mask.any():
                continue

            # Find contours for this class
            contours, _ = cv2.findContours(
                class_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw label on largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                # Get center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Get class name
                    class_name = self.seg_categories.get(class_id, f'Class {class_id}')

                    # Draw text with background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )

                    # Draw semi-transparent background
                    cv2.rectangle(result,
                                  (cx - text_width // 2 - 5, cy - text_height - baseline - 5),
                                  (cx + text_width // 2 + 5, cy + baseline),
                                  (255, 255, 255), -1)

                    # Draw text
                    cv2.putText(result, class_name,
                                (cx - text_width // 2, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return result

    def add_legend(self, image, mode='segmentation'):
        """Add color legend to the image"""
        legend_height = 200
        legend_width = 250
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

        y_offset = 20

        if mode == 'segmentation':
            cv2.putText(legend, 'Segmentation Classes:', (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25

            for class_id in range(1, 6):  # Skip background
                color = self.color_map[class_id]
                class_name = self.seg_categories[class_id]

                # Draw color box
                cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)
                cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), (0, 0, 0), 1)

                # Draw class name
                cv2.putText(legend, class_name, (35, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                y_offset += 25

        # Combine legend with image
        h, w = image.shape[:2]
        result = np.zeros((h, w + legend_width, 3), dtype=np.uint8)
        result[:h, :w] = image
        result[:legend_height, w:] = legend
        result[legend_height:, w:] = 255  # White background for remaining area

        return result

    def process_image(self, image_path, output_path):
        """Process single image with both detection and segmentation"""
        # Read image
        image = cv2.imread(str(image_path))

        # Detection
        det_results = self.detect_objects(image)
        det_image = self.draw_detection_boxes(image, det_results)

        # Segmentation
        seg_mask = self.segment_image(image, confidence_threshold = 0.9)
        seg_image = self.visualize_segmentation(image, seg_mask)
        seg_image_with_legend = self.add_legend(seg_image, mode='segmentation')

        # Combine results (side by side)
        combined = np.hstack([det_image, seg_image])

        # Save results
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path / f'det_{Path(image_path).name}'), det_image)
        cv2.imwrite(str(output_path / f'seg_{Path(image_path).name}'), seg_image_with_legend)
        # Combined (side by side)
        # Resize detection image to match segmentation with legend
        det_resized = cv2.resize(det_image, (seg_image_with_legend.shape[1], seg_image_with_legend.shape[0]))
        combined = np.vstack([det_resized, seg_image_with_legend])
        cv2.imwrite(str(output_path / f'combined_{Path(image_path).name}'), combined)

        print(f"Saved results to {output_path}")

    def process_video(self, video_path, output_path):
        """Process video with both detection and segmentation"""
        cap = cv2.VideoCapture(str(video_path))

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writers
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_det = cv2.VideoWriter(str(output_path / 'detection.mp4'), fourcc, fps, (width, height))
        out_seg = cv2.VideoWriter(str(output_path / 'segmentation.mp4'), fourcc, fps, (width + 250, height))
        out_combined = cv2.VideoWriter(str(output_path / 'combined.mp4'), fourcc, fps, (width + 250, height * 2))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detection
            det_results = self.detect_objects(frame)
            det_frame = self.draw_detection_boxes(frame, det_results)

            # Segmentation
            seg_mask = self.segment_image(frame, confidence_threshold = 0.5)
            seg_frame = self.visualize_segmentation(frame, seg_mask)
            seg_frame_with_legend = self.add_legend(seg_frame, mode='segmentation')

            # Combined
            det_resized = cv2.resize(det_frame, (seg_frame_with_legend.shape[1], seg_frame_with_legend.shape[0]))
            combined_frame = np.vstack([det_resized, seg_frame_with_legend])

            # Write frames
            out_det.write(det_frame)
            out_seg.write(seg_frame_with_legend)
            out_combined.write(combined_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        # Release resources
        cap.release()
        out_det.release()
        out_seg.release()
        out_combined.release()

        print(f"Video processing completed. Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Coral Detection and Segmentation Inference')
    parser.add_argument('--detection-model', type=str, default='train/result/detection/detect_best.pt',
                        help='Path to YOLOv8 detection model')
    parser.add_argument('--segmentation-model', type=str, default='train/result/segmentation/adam_3_stage_model_best.pt',
                        help='Path to DeepLabV3 segmentation model')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image or video file')
    parser.add_argument('--output', type=str, default='inference_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda/cpu)')

    args = parser.parse_args()

    # Initialize inference
    inferencer = UODInference(
        detection_model_path=args.detection_model,
        segmentation_model_path=args.segmentation_model,
        device=args.device
    )

    # Check if source is image or video
    source_path = Path(args.source)
    if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        inferencer.process_image(args.source, args.output)
    elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        inferencer.process_video(args.source, args.output)
    else:
        print(f"Unsupported file format: {source_path.suffix}")


if __name__ == '__main__':
    main()
