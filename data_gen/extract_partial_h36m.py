import pickle
import numpy as np

source_h36m_file = 'data/source_data/H36M.pkl'
target_h36m_file = 'data/source_data/H36M_Actions6.pkl'

all_action_names = ['Direction', 'Discuss', 'Eating', 'Greet', 'Phone',
               'Pose', 'Purchase', 'Sitting', 'SittingDown', 'Smoke',
               'Photo', 'Wait', 'Walk', 'WalkDog', 'WalkTwo']
all_action_numbered = [2, 3, 4, 5, 6,
                   7, 8, 9, 10, 11,
                   12, 13, 14, 15, 16]

action_to_extract = [2, 3, 4, 5, 6,
                     7, 8, 9]

action_to_extract = [2, 3, 4, 5, 6,
                     7]

# 1. 加载原始的 pkl 文件
with open(source_h36m_file, 'rb') as f:
    data = pickle.load(f)

data_train = data["train"]

data_train_joint_2d = data_train["joint_2d"]
data_train_confidence = data_train["confidence"]
data_train_joint3d_image = data_train["joint3d_image"]
data_train_camera_name = data_train["camera_name"]
data_train_source = data_train["source"]

new_data_train_joint_2d = []
new_data_train_confidence = []
new_data_train_joint3d_image = []
new_data_train_camera_name = []
new_data_train_source = []
for idx, source in enumerate(data_train_source):
    action_number = int(source[9:11])
    if action_number in action_to_extract:
        new_data_train_joint_2d.append(data_train_joint_2d[idx])
        new_data_train_confidence.append(data_train_confidence[idx])
        new_data_train_joint3d_image.append(data_train_joint3d_image[idx])
        new_data_train_camera_name.append(data_train_camera_name[idx])
        new_data_train_source.append(source)

new_data_train_joint_2d = np.array(new_data_train_joint_2d)
new_data_train_confidence = np.array(new_data_train_confidence)
new_data_train_joint3d_image = np.array(new_data_train_joint3d_image)
new_data_train_camera_name = np.array(new_data_train_camera_name, dtype='<U8')

new_data_train = {
    "joint_2d": new_data_train_joint_2d,
    "confidence": new_data_train_confidence,
    "joint3d_image": new_data_train_joint3d_image,
    "camera_name": new_data_train_camera_name,
    "source": new_data_train_source
}






data_test = data["test"]

data_test_joint_2d = data_test["joint_2d"]
data_test_confidence = data_test["confidence"]
data_test_joint3d_image = data_test["joint3d_image"]
data_test_joints_25d_image = data_test["joints_2.5d_image"]
data_test_25d_factor = data_test["2.5d_factor"]
data_test_camera_name = data_test["camera_name"]
data_test_action = data_test["action"]
data_test_source = data_test["source"]

new_data_test_joint_2d = []
new_data_test_confidence = []
new_data_test_joint3d_image = []
new_data_test_joints_25d_image = []
new_data_test_25d_factor = []
new_data_test_camera_name = []
new_data_test_action = []
new_data_test_source = []

for idx, source in enumerate(data_test_source):
    action_number = int(source[9:11])
    if action_number in action_to_extract:
        new_data_test_joint_2d.append(data_test_joint_2d[idx])
        new_data_test_confidence.append(data_test_confidence[idx])
        new_data_test_joint3d_image.append(data_test_joint3d_image[idx])
        new_data_test_joints_25d_image.append(data_test_joints_25d_image[idx])
        new_data_test_25d_factor.append(data_test_25d_factor[idx])
        new_data_test_camera_name.append(data_test_camera_name[idx])
        new_data_test_action.append(data_test_action[idx])
        new_data_test_source.append(source)

new_data_test_joint_2d = np.array(new_data_test_joint_2d)
new_data_test_confidence = np.array(new_data_test_confidence)
new_data_test_joint3d_image = np.array(new_data_test_joint3d_image)
new_data_test_joints_25d_image = np.array(new_data_test_joints_25d_image)
new_data_test_25d_factor = np.array(new_data_test_25d_factor)
new_data_test_camera_name = np.array(new_data_test_camera_name, dtype='<U8')

new_data_test = {
    "joint_2d": new_data_test_joint_2d,
    "confidence": new_data_test_confidence,
    "joint3d_image": new_data_test_joint3d_image,
    "joints_2.5d_image": new_data_test_joints_25d_image,
    "2.5d_factor": new_data_test_25d_factor,
    "camera_name": new_data_test_camera_name,
    "action": new_data_test_action,
    "source": new_data_test_source
}

new_data = {
    "train": new_data_train,
    "test": new_data_test
}

with open(target_h36m_file, 'wb') as f:
    pickle.dump(new_data, f)
