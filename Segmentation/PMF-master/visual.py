import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import yaml

def show_image(image, legend_patches):
    # 将 BGR 图像转换为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # 隐藏坐标轴

    # 添加图例
    # plt.legend(handles=legend_patches, loc='right', fontsize='small')
    plt.show()

def save_image(image, path):
    cv2.imwrite(path, image)


# 1. 读取点云数据
def load_point_cloud(sequence, frame, root_dir):
    bin_path = f"{root_dir}/sequences/{sequence:02d}/velodyne/{frame:06d}.bin"
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # 只使用XYZ坐标


# 2. 读取预测的语义标签
def load_label(sequence, frame, root_dir):
    label_path = f"{root_dir}/sequences/{sequence:02d}/labels/{frame:06d}.label"
    return np.fromfile(label_path, dtype=np.uint32) & 0xFFFF  # 只使用低16位

def load_prediction(sequence, frame, root_dir):
    label_path = f"{root_dir}/sequences/{sequence:02d}/predictions/{frame:06d}.label"
    return np.fromfile(label_path, dtype=np.uint32) & 0xFFFF  # 只使用低16位

# 3. 读取相机标定文件
def load_calibration(sequence, root_dir):
    calib_path = f"{root_dir}/sequences/{sequence:02d}/calib.txt"
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    # 从标定文件中读取P2和Tr_velo_to_cam矩阵
    P2 = np.array([float(value) for value in lines[2].strip().split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.array([float(value) for value in lines[4].strip().split(' ')[1:]]).reshape(3, 4)

    # 扩展 Tr_velo_to_cam 到 4x4 矩阵
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))

    return P2, Tr_velo_to_cam


# 4. 将点云投射到图像上
def project_lidar_to_image(point_cloud, predictions, image, P2, Tr_velo_to_cam):
    # 将点云从 LiDAR 坐标系转换到相机坐标系
    num_points = point_cloud.shape[0]
    point_cloud_hom = np.hstack((point_cloud, np.ones((num_points, 1))))  # (N, 4)
    point_cloud_cam = (Tr_velo_to_cam @ point_cloud_hom.T).T  # (N, 4)

    # 只保留 z > 0 的点（前方的点）
    valid_indices = point_cloud_cam[:, 2] > 0
    point_cloud_cam = point_cloud_cam[valid_indices]
    predictions = predictions[valid_indices]

    # 投影到图像平面
    point_cloud_2d = (P2 @ point_cloud_cam.T).T  # (N, 3)
    point_cloud_2d[:, 0] /= point_cloud_2d[:, 2]
    point_cloud_2d[:, 1] /= point_cloud_2d[:, 2]
    points_2d = point_cloud_2d[:, :2].astype(int)  # (N, 2)

    # 在图像上绘制预测标签
    for i in range(points_2d.shape[0]):
        u, v = points_2d[i]
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:  # 检查是否在图像范围内
            color = get_color(predictions[i])  # 根据标签获取颜色
            cv2.circle(image, (u, v), 1, color, -1)  # 在图像上绘制点

    return image


# 5. 根据标签返回颜色
def get_color(label):
    color_map = {
        0: [255, 255, 255],
        1: [0, 0, 255],
        2: [245, 150, 100],
        3: [245, 230, 100],
        4: [250, 80, 100],
        5: [150, 60, 30],
        6: [255, 0, 0],
        7: [180, 30, 80],
        8: [30, 30, 255],
        9: [200, 40, 255],
        10: [90, 30, 150],
        11: [255, 0, 255],
        12: [255, 150, 255],
        13: [75, 0, 75],
        14: [75, 0, 175],
        15: [0, 200, 255],
        16: [50, 120, 255],
        17: [0, 150, 255],
        18: [170, 255, 150],
        19: [0, 175, 0],
    }
    return color_map.get(label)

def class_map(learning_map, label):
    # map unused classes to used classes
    max_key = 0
    for k, v in learning_map.items():
        if k > max_key:
            max_key = k
    # +100 hack making lut bigger just in case there are unknown labels
    class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
    for k, v in learning_map.items():
        class_map_lut[k] = v

    return class_map_lut[label]

# 主函数
if __name__ == "__main__":
    # 设置路径
    root_dir = "" # 数据集根目录
    prediction_root_dir = ""
    attack_prediction_root_dir = ""
    config_path="./pc_processor/dataset/semantic_kitti/semantic-kitti.yaml"
    save_dir = "./vis_images"

    sequence = 8  # 序列号（如 00, 01, 02 等）
    frame = 1158 # 帧号（如 000000, 000001 等）
    save_tag = True

    ## 读取config
    if os.path.isfile(config_path):
        data_config = yaml.safe_load(open(config_path, "r"))

    # 读取数据
    point_cloud = load_point_cloud(sequence, frame, root_dir)  # 读取点云数据

    label = load_label(sequence, frame, root_dir)
    predictions = load_prediction(sequence, frame, prediction_root_dir)  # 读取预测标签
    attack_predictions = load_prediction(sequence, frame, attack_prediction_root_dir)

    learning_map = data_config["learning_map"]
    label = class_map(learning_map, label)
    predictions = class_map(learning_map, predictions)
    attack_predictions = class_map(learning_map, attack_predictions)

    P2, Tr_velo_to_cam = load_calibration(sequence, root_dir)  # 读取标定信息

    image_path = f"{root_dir}/sequences/{sequence:02d}/image_2/{frame:06d}.png"
    image = cv2.imread(image_path)  # 读取图像

    mapped_class_name = data_config["mapped_class_name"]
    legend_patches = []
    for idx in range(20):
        bgr = get_color(idx)
        rgb = [bgr[2], bgr[1], bgr[0]]
        name = mapped_class_name[idx]
        legend_patches.append(mpatches.Patch(color=np.array(rgb) / 255.0, label=name))


    # 投影并显示结果
    gt_path = f"{save_dir}/seq_{sequence:02d}_{frame:06d}_gt.png"
    projected_image = project_lidar_to_image(point_cloud, label, image, P2, Tr_velo_to_cam)
    show_image(projected_image, legend_patches)
    if save_tag:
        save_image(projected_image, gt_path)

    pred_path = f"{save_dir}/seq_{sequence:02d}_{frame:06d}_pred.png"
    projected_image = project_lidar_to_image(point_cloud, predictions, image, P2, Tr_velo_to_cam)
    show_image(projected_image, legend_patches)
    if save_tag:
        save_image(projected_image, pred_path)

    attack_pred_path = f"{save_dir}/seq_{sequence:02d}_{frame:06d}_attack_pred.png"
    projected_image = project_lidar_to_image(point_cloud, attack_predictions, image, P2, Tr_velo_to_cam)
    show_image(projected_image, legend_patches)
    if save_tag:
        save_image(projected_image, attack_pred_path)

