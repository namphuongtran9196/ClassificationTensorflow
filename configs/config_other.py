# Model config
model_path='./checkpoints/mobilenetv2/20220428-170008/weights/mobilenetv2.h5'
class_info_path= "./checkpoints/mobilenetv2/20220428-170008/weights/class_info.json"
input_shape = (224, 224, 3)
# Dataset config
train_dataset = './data/dataset/train'
val_dataset = './data/dataset/val'
# Optimizer config
learning_rate = 1e-2

# Training config
checkpoints_dir = './checkpoints'
batch_size=128
epochs = 10




