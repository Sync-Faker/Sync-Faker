# Attacks on the PMF Model

## Preparing Semantic KITTI dataset and Install environment
Just refer to the original github repository of PMF (https://github.com/ICEORY/PMF).

## Without Attack
### Data preprocessing
1. Enter the tasks/process_semantickitti_fov directory and modify the dataset path src_root in the create_fov_dataset.py file to the actual path of semantic-kitti

2. Run the following command to build the semantic-kitti-fov dataset:

    python create_fov_dataset.py

### Model Inference
Please refer to the inference steps of the model in PMF's github repository

## With Attack
### Prepare the attack dataset
1. Please modify the source_root in the file to the path of the original dataset and the target_root to the path of the dataset with the attack.
2. run the generate_attack_kitti_datasets.py to generate the datasets with attack.

    python generate_attack_kitti_datasets.py

### Data preprocessing
1. Enter the tasks/process_semantickitti_fov directory and modify the dataset path src_root in the create_fov_dataset.py file to the actual path of semantic-kitti with attack

2. Run the following command to build the semantic-kitti-fov dataset:

    python create_fov_dataset.py

### Model Inference
Just refer to the inference steps of the model in PMF's github repository

## Visualize the results
1. Modify variables in visual.py file
    root_dir: Path to the original Semantic KITTI dataset
    prediction_root_dir: The path of the model's prediction results
    attack_prediction_root_dir: The path of the model's prediction results after being attacked

2. run the visual.py
    python visual.py