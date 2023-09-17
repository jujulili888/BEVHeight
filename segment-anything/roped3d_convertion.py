import os
import cv2
import csv
import glob

import torch
import torch.nn.functional as F

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Segmentix:
    def resize_mask(self, ref_mask: np.ndarray, longest_side: int = 256):
        """
        Resize an image to have its longest side equal to the specified value.
        Args:
            ref_mask (np.ndarray): The image to be resized.
            longest_side (int, optional): The length of the longest side after resizing. Default is 256.
        Returns:
            tuple[np.ndarray, int, int]: The resized image and its new height and width.
        """
        height, width = ref_mask.shape[:2]
        if height > width:
            new_height = longest_side
            new_width = int(width * (new_height / height))
        else:
            new_width = longest_side
            new_height = int(height * (new_width / width))
        return (cv2.resize(ref_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST),
                new_height,
                new_width,
        )

    def pad_mask(
        self,
        ref_mask: np.ndarray,
        new_height: int,
        new_width: int,
        pad_all_sides: bool = False,
    ) -> np.ndarray:
        """
        Add padding to an image to make it square.
        Args:
            ref_mask (np.ndarray): The image to be padded.
            new_height (int): The height of the image after resizing.
            new_width (int): The width of the image after resizing.
            pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.
        Returns:
            np.ndarray: The padded image.
        """
        pad_height = 256 - new_height
        pad_width = 256 - new_width
        if pad_all_sides:
            padding = (
                (pad_height // 2, pad_height - pad_height // 2),
                (pad_width // 2, pad_width - pad_width // 2),
            )
        else:
            padding = ((0, pad_height), (0, pad_width))
        # Padding value defaults to '0' when the `np.pad`` mode is set to 'constant'.
        return np.pad(ref_mask, padding, mode="constant")

    def reference_to_sam_mask(
        self, ref_mask: np.ndarray, threshold: int = 127, pad_all_sides: bool = False
    ) -> np.ndarray:
        """
        Convert a grayscale mask to a binary mask, resize it to have its longest side equal to 256, and add padding to make it square.
        Args:
            ref_mask (np.ndarray): The grayscale mask to be processed.
            threshold (int, optional): The threshold value for the binarization. Default is 127.
            pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.
        Returns:
            np.ndarray: The processed binary mask.
        """
        # Convert a grayscale mask to a binary mask.
        # Values over the threshold are set to 1, values below are set to -1.
        ref_mask = np.clip((ref_mask > threshold) * 2 - 1, -1, 1)

        # Resize to have the longest side 256.
        resized_mask, new_height, new_width = self.resize_mask(ref_mask)

        # Add padding to make it square.
        square_mask = self.pad_mask(resized_mask, new_height, new_width, pad_all_sides)

        # Expand SAM mask's dimensions to 1xHxW (1x256x256).
        return np.expand_dims(square_mask, axis=0)

def get_mesh_coordinates(image_size, stride=1):
    height, width = image_size[0], image_size[1]
    pixel_coordinates = []
    for y in np.arange(0, height, stride):
        for x in np.arange(0, width, stride):
            pixel_coordinates.append([x, y])
    pixel_coordinates = np.array(pixel_coordinates).astype(np.int)
    return pixel_coordinates

def get_bbox_mask_coordinates(bboxes, image_size, num):
    coordinates = []
    height, width = image_size[0], image_size[1]
    for i in range(bboxes.shape[0]): 
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, width - 1),  min(ymax, height - 1)
        bbox = [xmin, ymin, xmax, ymax]

        size = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        if size < 5: continue
        stride = int(size / num)
        for y in np.arange(int(bbox[1]), int(bbox[3]), stride):
            for x in np.arange(int(bbox[0]), int(bbox[2]), stride):
                coordinates.append([x, y])
    return np.array(coordinates)

def load_annos(label_path):
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                    'dl', 'lx', 'ly', 'lz', 'ry']
    bboxes, labels = [], []
    with open(label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            name = row["type"]
            dim = [float(row['dh']), float(row['dw']), float(row['dl'])]
            bbox = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]   #2D检测框位置
            if sum(bbox) == 0:
                continue
            bboxes.append(bbox)
            labels.append(name)
    bboxes = np.array(bboxes).astype(np.int32)
    return bboxes, labels

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def visualize(masks, bboxes, labels, image, filename):
    os.makedirs(os.path.join("output", "object_mask"), exist_ok=True)
    os.makedirs(os.path.join("output", "road_mask"), exist_ok=True)
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    mask_image = np.zeros((int(1080 * 0.8), int(1920 * 0.8), 1))
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        mask = mask.cpu().numpy()
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w, 1).astype(np.int8)
        mask_image += mask
    for box, label in zip(bboxes, labels):
        show_box(box, plt.gca(), label)
    plt.axis('off')
    plt.savefig(
        filename, 
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    return mask_image

def mask_inference(predictor, point_coords, point_labels, bboxes, mask_input, image):
    predictor.set_image(image)
    if point_coords is not None:
        assert (
            point_labels is not None
        ), "point_labels must be supplied if point_coords is supplied."
        point_coords = predictor.transform.apply_coords(point_coords, predictor.original_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=predictor.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=predictor.device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    else:
        coords_torch, labels_torch = None, None
    if bboxes is not None:
        bboxes  = torch.tensor(bboxes)
        transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2]).to(device)
        transformed_boxes = transformed_boxes.to(device)
    else:
        transformed_boxes = None
    if mask_input is not None:
        if isinstance(mask_input, np.ndarray):
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        else:
            mask_input_torch = mask_input
    else:
        mask_input_torch = None

    masks, _, low_res_masks, embeddings = predictor.predict_torch(
        point_coords = coords_torch,
        point_labels = labels_torch,
        boxes = transformed_boxes,
        mask_input = mask_input_torch,
        multimask_output = False,
    )
    return masks, low_res_masks, embeddings

def init_hash_map(mask):
    hash_map = dict()
    for mask_file in os.listdir(mask):
        mask_list = mask_file.split('_')
        hash_map[mask_list[0]] = "_".join(mask_list)
    return hash_map

def load_road_mask(calib_file):
    calib_file = calib_file.replace("Rope3D_0083", "Rope3D")
    calib_file = calib_file.replace("data", "root")
    calib = open(calib_file).readlines()
    calib = calib[0].split(' ')[1]
    hash_map = init_hash_map("mask")
    road_mask_path = os.path.join("mask", hash_map[calib])
    road_mask = cv2.imread(road_mask_path, 0) / 255.0
    road_mask = cv2.resize(road_mask, (int(road_mask.shape[1] * 0.8), int(road_mask.shape[0] * 0.8)))
    return road_mask

def save_embeddings(embeddings, npy_path):
    if len(embeddings.shape) == 3:
         embeddings = torch.unsqueeze(embeddings, dim=0)
    embeddings = F.interpolate(embeddings, (54, 96), mode="bilinear", align_corners=False)
    embeddings = embeddings.cpu().numpy()
    np.save(npy_path, embeddings)

def save_embeddings_compressed(npz_path, obj_embeds, road_embeds):
    if len(obj_embeds.shape) == 3:
        obj_embeds = torch.unsqueeze(obj_embeds, dim=0)
    # obj_embeds = F.interpolate(obj_embeds, (54, 96), mode="bilinear", align_corners=False)
    obj_embeds = obj_embeds.cpu().numpy()
    road_embeds = road_embeds.cpu().numpy()

    np.savez_compressed(npz_path, obj_embeds=obj_embeds, road_embeds=road_embeds)
    # np.savez_compressed(npz_path, obj_embeds=obj_embeds)

def get_road_embbeds(calib_file, image, index, masks, bboxes, labels, visual=False):
    point_coords, point_labels = None, None
    road_mask = load_road_mask(calib_file)
    road_mask = torch.as_tensor(road_mask, dtype=torch.int, device="cuda")
    road_mask = road_mask.view(1, 1, road_mask.shape[0], road_mask.shape[1])
    road_image = (road_mask - masks.int()).cpu().numpy()[0, 0]
    segmentix = Segmentix()
    sam_road_image = segmentix.reference_to_sam_mask(road_image * 255)
    box_promt = np.array([[0, 0, image.shape[1] - 1, image.shape[0] - 1]])
    masks, _, embeddings = mask_inference(predictor, point_coords, point_labels, box_promt, sam_road_image, image)
    if visual:
        visualize(masks, bboxes, labels, image, os.path.join("output", "road_mask", index + ".jpg"))
    return embeddings

def get_object_embbeds_one_mask(image, index, mask_image, low_res_masks, bboxes, labels, visual=False):
    low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)
    bbp_coords = get_bbox_mask_coordinates(bboxes, image.shape[:2], num=5)
    bbp_labels = mask_image[bbp_coords[:, 1], bbp_coords[:, 0]]
    point_coords = get_mesh_coordinates(image.shape[:2], stride=16)
    point_labels = mask_image[point_coords[:, 1], point_coords[:,0]]
    point_coords = np.concatenate((point_coords, bbp_coords), axis=0)
    point_labels = np.concatenate((point_labels, bbp_labels), axis=0)
    masks, _, embeddings = mask_inference(predictor, point_coords, point_labels, None, low_res_masks, image)
    if visual:
        visualize(masks, bboxes, labels, image, os.path.join("output", "object_mask", index + ".jpg"))
    return embeddings

def sam_inference(predictor, src_dir, sub_img_path, index):
    label_path = os.path.join(src_dir, "label_2")
    calib_path = os.path.join(src_dir, "calib")
    img_file = os.path.join(sub_img_path, index + ".jpg")
    label_file = os.path.join(label_path, index + ".txt")
    calib_file = os.path.join(calib_path, index + ".txt")
    
    bboxes, labels = load_annos(label_file)
    bboxes = 0.8 * bboxes
    point_coords, point_labels = None, None
    if bboxes.shape[0] == 0: return
    save_file = os.path.join("object_embeddings", index + ".npy")
    if os.path.exists(save_file) and os.path.getsize(save_file) > 0: return

    image = cv2.imread(img_file)
    image = cv2.resize(image, (int(image.shape[1] * 0.8), int(image.shape[0] * 0.8)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks, low_res_masks, embeddings = mask_inference(predictor, point_coords, point_labels, bboxes, None, image)
    masks, _ = torch.max(masks, dim=0)
    masks = masks.cpu().numpy().astype(np.uint8) * 255
    cv2.imwrite(os.path.join("mask_image", index + ".jpg"), masks[0])

    mask_image = visualize(masks, bboxes, labels, image, os.path.join("output", "object_mask", index + ".jpg"))
    # obj_embeds, _ = torch.max(embeddings, dim=0)
    obj_embeds = torch.mean(embeddings, dim=0)
    save_embeddings(embeddings, os.path.join("object_embeddings", index + ".npy"))
    road_embeds =  get_road_embbeds(calib_file, image, index, masks, bboxes, labels, visual=True)
    # mask_image = torch.sum(masks, dim=0, keepdim=True)[0, 0].cpu().numpy()
    # embeddings = get_object_embbeds_one_mask(image, index, mask_image, low_res_masks, bboxes, labels, visual=False)
    # save_embeddings(embeddings, os.path.join("object_embeddings_one_mask", index + ".npy"))
    save_embeddings_compressed(os.path.join("object_embeddings", index + ".npz"), obj_embeds, road_embeds)

    
def generate_info_rope3d(predictor, rope3d_root, split='train'):
    if split == 'train':
        src_dir = os.path.join(rope3d_root, "training")
        img_path = ["training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
    else:
        src_dir = os.path.join(rope3d_root, "validation")
        img_path = ["validation-image_2"]

    split_txt = os.path.join(src_dir, "train.txt" if split=='train' else 'val.txt')
    idx_list = [x.strip() for x in open(split_txt).readlines()]
    idx_list_valid = []
    for index in idx_list:
        for sub_img_path in img_path:
            img_file = os.path.join(rope3d_root, sub_img_path, index + ".jpg")
            if os.path.exists(img_file):
                idx_list_valid.append((sub_img_path, index))
                break

    for idx in tqdm(range(len(idx_list_valid))):
        sub_img_path, index = idx_list_valid[idx]
        sam_inference(predictor, src_dir, os.path.join(rope3d_root, sub_img_path), index)

if __name__ == "__main__":
    rope3d_root = "/data/Rope3D_0083"
    sam_checkpoint, sam_hq_checkpoint = "./sam_vit_h_4b8939.pth", "./sam_hq_vit_h.pth"
    use_sam_hq = False
    device = "cuda"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("object_embeddings", exist_ok=True)
    os.makedirs("object_embeddings_one_mask", exist_ok=True)
    os.makedirs("road_embeddings", exist_ok=True)

    hash_map = init_hash_map("mask")
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    generate_info_rope3d(predictor, rope3d_root, split='train')
    generate_info_rope3d(predictor, rope3d_root, split='val')
