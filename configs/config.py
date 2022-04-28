# Model config
backbone = 'mobilenetv2'
preprocessing = True
input_shape = (224, 224, 3)
dropout = 0.2
# Dataset config
train_dataset = './dataset/train'
val_dataset = './dataset/val'
# Optimizer config
learning_rate = 1e-2

# Training config
checkpoints_dir = './checkpoints'
old_checkpoints_path = None
epochs = 20




