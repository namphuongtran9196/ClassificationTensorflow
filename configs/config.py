# Model config
backbone = 'mobilenetv2'
preprocessing = True
input_shape = (224, 224, 3)
dropout = 0.2
# Dataset config
train_dataset = './data/dataset/train'
val_dataset = './data/dataset/val'
# Optimizer config
learning_rate = 1e-2

# Training config
checkpoints_dir = './checkpoints'
old_checkpoints_path = None
class_info_path = None
batch_size=64
epochs = 20




