import cv2
import argparse

from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(description='detection inference')

    parser.add_argument('--video_path', type=str, required=True, help='video path')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--output_path', type=str, default="output_result.mp4", help='output directory')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    model = YOLO(args.model_path)

    result = model.predict(source=args.video_path, stream=True, conf=args.conf, save=False)

    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    for r in result:
        annotated_frame = r.plot()
        out.write(annotated_frame)

    out.release()