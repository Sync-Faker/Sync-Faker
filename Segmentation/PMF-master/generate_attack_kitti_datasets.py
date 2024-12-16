import os
import shutil

# 定义源目录和目标目录
source_root = "...dataset/SemanticKITTI/dataset/sequences/08"
target_root = "...dataset/SemanticKITTI_Attack/dataset/sequences/08"

print("rename images...")
source_dir = os.path.join(source_root,'image_3')
target_dir = os.path.join(target_root,'image_3')

# 检查目标目录是否存在，如果不存在则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

file_names = os.listdir(source_dir)
file_names.sort()
for idx in range(len(file_names)):
    source_file_name = file_names[idx]
    file_path = os.path.join(source_dir, source_file_name)

    # 定义新的文件名，这里可以根据需要进行修改
    new_filename = file_names[idx-2]
    target_file_path = os.path.join(target_dir, new_filename)

    # 复制并重命名文件到目标目录
    shutil.copyfile(file_path, target_file_path)

print("copy labels folder ...")
# copy image and calib files
src_label_folder = os.path.join(source_root, "labels")
dst_label_folder = os.path.join(target_root, "labels")
shutil.copytree(src_label_folder, dst_label_folder)


print("copy velodyne folder ...")
# copy image and calib files
src_velodyne_folder = os.path.join(source_root, "velodyne")
dst_velodyne_folder = os.path.join(target_root, "velodyne")
shutil.copytree(src_velodyne_folder, dst_velodyne_folder)

target_files = ["calib.txt", "poses.txt", "times.txt"]
print("copy calib files ...")
for f_name in target_files:
    src_file_path = os.path.join(source_root, f_name)
    dst_file_path = os.path.join(target_root, f_name)
    shutil.copyfile(src_file_path, dst_file_path)