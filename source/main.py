import argparse
import glob
import os

import cv2
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from tqdm import tqdm
from ultralytics import YOLO

# # SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features="superpoint").eval().cuda()  # load the matcher


def feature_matching(template_path, target_image_path):
    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image0 = load_image(template_path, resize=50).cuda()
    image1 = load_image(target_image_path, thresh=False).cuda()

    # extract local features
    feats0 = extractor.extract(
        image0
    )  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)

    # match the features
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = (
        feats0["keypoints"],
        feats1["keypoints"],
        matches01["matches"],
    )
    m_kpts0, _ = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return len(m_kpts0)


def inference_feature_matching(image_path, hero_thumbnails):
    max_points = {}
    for thumbnail in tqdm(hero_thumbnails):
        hero_name = thumbnail.replace("./data/thumbnail/", "").replace(".png", "")
        match_points = feature_matching(thumbnail, image_path)
        max_points[hero_name] = match_points
    max_points = dict(sorted(max_points.items(), key=lambda item: item[1]))
    max_key = max(max_points, key=max_points.get)
    return max_key, max_points[max_key]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hero recognition program.")
    parser.add_argument(
        "--input",
        help="Path to the test_images/ folder.",
        default="./data/test_data/test_images",
    )
    parser.add_argument("--output", help="Path to the output.txt.", default="./")
    args = parser.parse_args()

    test_path = os.path.join(args.input, "*")
    output_path = os.path.join(args.output, "output.txt")
    print(f"Path to test_images/ folder: {test_path}")
    print(f"Path to output.txt: {output_path}")

    test_data = glob.glob(test_path)
    output_hero_names = open(output_path, "w")
    hero_thumbnails = glob.glob("./data/thumbnail/*")  # Reference data

    # Load a pretrained YOLOv8n model
    model = YOLO("./yolo_detection/code/runs/detect/train/weights/best.pt")

    for sample in tqdm(test_data):
        results = model.predict(
            sample,
            # save=True,
            conf=0.25,
            verbose=False,
        )  # predict on an image
        for r in results:
            # Choose box with lowest x coordinate
            data = r.boxes.data
            x_ele = data[:, 0].reshape(1, -1)
            min_elements, min_idx = torch.min(x_ele, dim=1)
            target_box = data[min_idx.cpu()][0]

            # Crop target box with target hero
            bb, cls, conf = target_box[:4], target_box[4], target_box[5]
            bb = [int(coor) for coor in bb]
            img = cv2.imread(sample)
            hero_img = img[bb[1] : bb[1] + bb[3], bb[0] : bb[0] + bb[2]]
            hero_img_path = f"./temp_hero_img.jpg"
            cv2.imwrite(hero_img_path, hero_img)

            # Match target box with hero's thumbnails
            predicted_name, match_keypoints = inference_feature_matching(
                hero_img_path, hero_thumbnails
            )
            output_hero_names.write(f"{sample}\t{predicted_name}\n")
    output_hero_names.close()
