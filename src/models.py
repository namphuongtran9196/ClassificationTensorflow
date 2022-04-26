import tensorflow as tf

from tensorflow.keras import layers, Model

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet50
# from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input as preprocess_input_resnet50v2
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_input_mobilenetv2

def build_resnet50_classification(num_classes,input_shape=(224,224,3),dropout=0.2,preprocessing=True):
    inputs = layers.Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense(num_classes)
    dropout_layer = layers.Dropout(0.2)
    if preprocessing:
        x = preprocess_input_resnet50(inputs)
    x = base_model(x)
    x = global_average_layer(x)
    x = dropout_layer(x)
    x = prediction_layer(x)
    return Model(inputs, x, name="Resnet50_Classification")

if __name__ == "__main__":
    model = build_resnet50_classification(num_classes=2)
    model.summary()