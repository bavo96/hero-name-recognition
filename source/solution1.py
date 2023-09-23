# SOLUTION 1: Feature matching + choose highest keypoints

import glob
from pathlib import Path

from lightglue import DISK, LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
from matplotlib import pyplot as plt
from tqdm import tqdm

hero_thumbnails = glob.glob("./data/thumbnail/*png")
test_data = open("./data/test_data/test.txt", "r").readlines()

# # SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features="superpoint").eval().cuda()  # load the matcher


def feature_matching(template_path, target_image_path):
    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image0 = load_image(template_path, resize=50).cuda()

    image1 = load_image(target_image_path).cuda()
    new_width = int(image1.shape[2] / 3)
    image1 = image1[:, 0 : image1.shape[1], 0:new_width]

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
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Unblock below code to get the original keypoints and the visualization code
    # hero_name = template_path.replace("./data/thumbnail/", "").replace(".png", "")
    # axes = viz2d.plot_images([image0, image1])
    # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
    # plt.savefig(f"./result/{hero_name}_matches.jpg")
    # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(
    #     matches01["prune1"]
    # )
    # viz2d.plot_images([image0, image1])
    # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
    # plt.savefig(f"./result/{hero_name}_keypoints.jpg")
    return len(m_kpts0)


def inference(image_path):
    max_points = {}
    for thumbnail in tqdm(hero_thumbnails):
        hero_name = thumbnail.replace("./data/thumbnail/", "").replace(".png", "")
        match_points = feature_matching(thumbnail, image_path)
        max_points[hero_name] = match_points
    max_points = dict(sorted(max_points.items(), key=lambda item: item[1]))
    max_key = max(max_points, key=max_points.get)
    return max_key, max_points[max_key]


if __name__ == "__main__":
    # Path("./result/").mkdir(parents=True, exist_ok=True)

    true_prediction = 0
    for sample in tqdm(test_data):
        filename, label = sample.split()
        image_path = f"./data/test_data/test_images/{filename}"
        predicted_name, points = inference(image_path)
        print(f"Target image: {filename}. Hero's name: {label}")
        print(f"Prediction: {predicted_name}. Keypoints: {points}")
        if label == predicted_name:
            true_prediction += 1

    accuracy = true_prediction / len(test_data)
    print(f"True predictions:{true_prediction}")
    print(f"Total samples:{len(test_data)}")
    print(f"Accuracy: {accuracy:.2f}%")
