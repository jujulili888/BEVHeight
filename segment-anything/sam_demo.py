import os
import cv2
import csv

import torch
import torch.nn.functional as F

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)

import numpy as np
import matplotlib.pyplot as plt

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
            if sum(dim) == 0:
                continue
            bboxes.append(bbox)
            labels.append(name)

    bboxes = np.array(bboxes).astype(np.int32)
    return bboxes, labels

def mask_inference(predictor, point_coords, point_labels, bboxes, mask_input, image):
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

    masks, _, low_res_masks = predictor.predict_torch(
        point_coords = coords_torch,
        point_labels = labels_torch,
        boxes = transformed_boxes,
        mask_input = mask_input,
        multimask_output = False,
    )
    return masks, low_res_masks

def sam_combine_tools(predictor, bboxes, src_image, dest_image):
    masks, distill = mask_inference(predictor, bboxes, src_image)
    mask_image = np.zeros((int(1080 * 0.8), int(1920 * 0.8), 1))
    for mask in masks:
        mask = mask.cpu().numpy()
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w, 1).astype(np.int8)
        mask_image += mask
        print(np.min(mask), np.max(mask))
    dest_image = dest_image * (1 - mask_image) + src_image * mask_image        
    return dest_image, masks

def sam_init(sam_checkpoint, device="cuda"):
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return predictor

def visualize(masks, bboxes, image, filename):
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
        stride = int(size / num)
        for y in np.arange(int(bbox[1]), int(bbox[3]), stride):
            for x in np.arange(int(bbox[0]), int(bbox[2]), stride):
                coordinates.append([x, y])
    return np.array(coordinates)

if __name__ == "__main__":
    sam_checkpoint, sam_hq_checkpoint = "./sam_vit_h_4b8939.pth", "./sam_hq_vit_h.pth"
    device = "cuda"
    output_dir = "output"
    image_path = "/data/Rope3D_0083/training-image_2a/145040_fa2sd4a06W152AIR_420_1626155124_1626155723_166_obstacle.jpg"
    label_path = "/data/Rope3D_0083/training/label_2/145040_fa2sd4a06W152AIR_420_1626155124_1626155723_166_obstacle.txt"
    calib_path = "/data/Rope3D/training/calib/145040_fa2sd4a06W152AIR_420_1626155124_1626155723_166_obstacle.txt"

    dest_image_path = "/data/Rope3D_0083/training-image_2a/62533_fa2sd4a13East154_420_1625815958_1625816709_239_obstacle.jpg"

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # initialize SAM
    calib = open(calib_path).readlines()
    calib = calib[0].split(' ')[1]
    print("calib: ", calib)
    road_mask_path = os.path.join("mask", "2162.827939_camera3_mask.jpg")
    print("road_mask_path: ", road_mask_path)
    road_mask = cv2.imread(road_mask_path, 0) / 255.0
    road_mask = cv2.resize(road_mask, (int(road_mask.shape[1] * 0.8), int(road_mask.shape[0] * 0.8)))
    print("output/oad_mask: ", road_mask.shape)
    cv2.imwrite("output/road_mask.jpg", road_mask)

    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    bboxes, labels = load_annos(label_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(image.shape[1] * 0.8), int(image.shape[0] * 0.8)))
    image_demo = image.copy()

    bboxes = 0.8 * bboxes
    point_coords, point_labels = None, None
    masks, low_res_masks = mask_inference(predictor, point_coords, point_labels, bboxes, None, image)
    mask_image_1 = visualize(masks, bboxes, image, os.path.join("output", "grounded_sam_output.jpg"))

    masks = torch.sum(masks, dim=0, keepdim=True)
    road_mask = torch.as_tensor(road_mask, dtype=torch.int, device=device)
    road_mask = road_mask.view(1, 1, road_mask.shape[0], road_mask.shape[1])
    road_mask = (road_mask - masks)
    road_image = road_mask.cpu().numpy()[0, 0]
    road_image = masks.cpu().numpy()[0, 0]

    visualize(road_mask, bboxes, image_demo, os.path.join("output", "road_mask.jpg"))

    cv2.imwrite("output/mask_image.jpg", mask_image_1*255)
    low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)

    bbp_coords = get_bbox_mask_coordinates(bboxes, image.shape[:2], num=8)
    print(np.max(bbp_coords[:, 1]), np.max(bbp_coords[:, 0]), road_image.shape)
    bbp_labels = road_image[bbp_coords[:, 1], bbp_coords[:, 0]]

    mask_image = np.zeros((int(1080 * 0.8), int(1920 * 0.8), 1))
    mask_image[bbp_coords[:, 1], bbp_coords[:, 0], 0] = bbp_labels
    cv2.imwrite("output/bbp_labels.jpg", mask_image*255)

    point_coords = get_mesh_coordinates(image.shape[:2], stride=15)
    point_labels = road_image[point_coords[:, 1], point_coords[:,0]]

    point_coords = np.concatenate((point_coords, bbp_coords), axis=0)
    point_labels = np.concatenate((point_labels, bbp_labels), axis=0)
    
    box_promt = np.array([[0, 0, image.shape[1] - 1, image.shape[0] - 1]])
    masks, low_res_masks_temp = mask_inference(predictor, point_coords, point_labels, box_promt, low_res_masks, image)
    visualize(masks, bboxes, image_demo, os.path.join("output", "grounded_sam_output1.jpg"))

'''
if __name__ == "__main__":
    sam_checkpoint, sam_hq_checkpoint = "./sam_vit_h_4b8939.pth", "./sam_hq_vit_h.pth"
    device = "cuda"
    output_dir = "output"
    image_path = "/data/Rope3D_0083/training-image_2a/145040_fa2sd4a06W152AIR_420_1626155124_1626155723_166_obstacle.jpg"
    label_path = "/data/Rope3D_0083/training/label_2/145040_fa2sd4a06W152AIR_420_1626155124_1626155723_166_obstacle.txt"
    calib_path = "/data/Rope3D/training/calib/145040_fa2sd4a06W152AIR_420_1626155124_1626155723_166_obstacle.txt"

    dest_image_path = "/data/Rope3D_0083/training-image_2a/62533_fa2sd4a13East154_420_1625815958_1625816709_239_obstacle.jpg"

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # initialize SAM
    calib = open(calib_path).readlines()
    calib = calib[0].split(' ')[1]
    print("calib: ", calib)
    road_mask_path = os.path.join("mask", "2162.827939_camera3_mask.jpg")
    print("road_mask_path: ", road_mask_path)
    road_mask = cv2.imread(road_mask_path, 0) / 255.0
    road_mask = cv2.resize(road_mask, (int(road_mask.shape[1] * 0.8), int(road_mask.shape[0] * 0.8)))
    print("output/oad_mask: ", road_mask.shape)
    cv2.imwrite("output/road_mask.jpg", road_mask)

    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    bboxes, labels = load_annos(label_path)
    image = cv2.imread(image_path)
    dest_image = cv2.imread(dest_image_path)
    image = cv2.resize(image, (int(image.shape[1] * 0.8), int(image.shape[0] * 0.8)))

    bboxes = 0.8 * bboxes
    point_coords, point_labels = None, None
    masks, low_res_masks = mask_inference(predictor, point_coords, point_labels, bboxes, None, image)

    masks = torch.sum(masks, dim=0, keepdim=True)
    road_mask = torch.as_tensor(road_mask, dtype=torch.int, device=device)
    road_mask = road_mask.view(1, 1, road_mask.shape[0], road_mask.shape[1])
    # masks = torch.cat((road_mask, masks), dim=0)
    masks = (road_mask - masks)
    print("masks: ", masks.shape, road_mask.shape)

    mask_image_1 = visualize(masks, bboxes, image, os.path.join("output", "grounded_sam_output.jpg"))
    print("1. low_res_mask_demo: ", low_res_masks.shape, low_res_masks.shape, torch.max(low_res_masks), mask_image_1.shape)
    cv2.imwrite("output/mask_image.jpg", mask_image_1*255)
    low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)

    # bbp_coords = ((bboxes[:, :2] + bboxes[:, 2:]) / 2).astype(np.int32)
    # bbp_labels = np.ones((bbp_coords.shape[0])).astype(np.int32)

    bbp_coords = get_bbox_mask_coordinates(bboxes, num=3)
    bbp_labels = mask_image_1[bbp_coords[:, 1], bbp_coords[:, 0], 0]

    mask_image = np.zeros((int(1080 * 0.8), int(1920 * 0.8), 1))
    mask_image[bbp_coords[:, 1], bbp_coords[:, 0], 0] = bbp_labels
    cv2.imwrite("output/bbp_labels.jpg", mask_image*255)

    pixel_coordinates = get_mesh_coordinates(image.shape[:2], stride=16)
    pixel_coordinates_labels = mask_image_1[pixel_coordinates[:, 1], pixel_coordinates[:,0]]
    point_coords = pixel_coordinates
    point_labels = pixel_coordinates_labels[:, 0]


    point_coords = np.concatenate((point_coords, bbp_coords), axis=0)
    point_labels = np.concatenate((point_labels, bbp_labels), axis=0)
    
    box_promt = np.array([[0, 0, image.shape[1] - 1, image.shape[0] - 1]])
    masks, low_res_masks_temp = mask_inference(predictor, point_coords, point_labels, None, low_res_masks, image)
    mask_image_2 = visualize(masks, bboxes, image, os.path.join("output", "grounded_sam_output1.jpg"))
   
    # 1 - 0 = 1    false positive  --> background
    # 0 - 1  = -1  false negative  ---> forground

    mask_image_1, mask_image_2 = mask_image_1.astype(np.int), mask_image_2.astype(np.int)
    delta_image = (mask_image_2 - mask_image_1)
    coors = get_mesh_coordinates(image.shape[:2], stride=4)
    coors_labels = delta_image[coors[:, 1], coors[:,0], 0]
    cv2.imwrite("output/FP.jpg", np.clip(delta_image, 0, 1)*255)
    cv2.imwrite("output/FN.jpg", -1 * np.clip(delta_image, -1, 0)*255)

    back_coors = coors[coors_labels == 1]
    back_labels = 1 - coors_labels[coors_labels == 1]
    fore_coors = coors[coors_labels == -1]
    fore_labels = -1 * coors_labels[coors_labels == -1]

    print("point_coords: ", point_coords.shape, point_labels.shape)
    print("fore_coors: ", fore_coors.shape, fore_labels.shape)
    print("fore_coors: ", back_coors.shape, back_labels.shape)

    point_coords = np.concatenate((fore_coors, back_coors), axis=0)
    point_labels = np.concatenate((fore_labels, back_labels), axis=0)

    print("coors_labels: ", point_coords.shape, back_labels, point_labels.shape)

    masks, low_res_masks = mask_inference(predictor, point_coords, point_labels, None, low_res_masks, image)
    mask_image_2 = visualize(masks, bboxes, image, os.path.join("output", "grounded_sam_output2.jpg"))
    print("2. low_res_mask_demo: ", low_res_masks.shape, low_res_masks.shape, torch.max(low_res_masks))    
'''