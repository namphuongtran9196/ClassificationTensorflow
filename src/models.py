import tensorflow as tf

from tensorflow.keras import layers, Model

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input as preprocess_input_resnet50v2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_input_mobilenetv2

def build_classification_model(num_classes,input_shape=(224,224,3),backbone="mobilenetv2",dropout=0.2,preprocessing=True):
    # input image with input_shape
    inputs = layers.Input(shape=input_shape)
    # pass image through backbone, freeze backbone layers
    if backbone.lower() == "resnet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        base_model.trainable = False
        x = preprocess_input_resnet50(inputs) if preprocessing else inputs
    elif backbone.lower() == "resnet50v2":
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_tensor=inputs)
        base_model.trainable = False
        x = preprocess_input_resnet50v2(inputs) if preprocessing else inputs
    elif backbone.lower() == "mobilenetv2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
        base_model.trainable = False
        x = preprocess_input_mobilenetv2(inputs) if preprocessing else inputs
    else:
        raise ValueError("Backbone {} is not supported".format(backbone))
    x = base_model(x)
    # add global average pooling layer
    x = layers.GlobalAveragePooling2D(name='global_average_layer')(x)
    # add dropout layer
    x = layers.Dropout(dropout,name='dropout_layer')(x)
    # add multi perceptron layer
    x = layers.Dense(num_classes,name='prediction_logits_layer')(x)
    # add softmax activation layer
    x = layers.Activation("softmax",name='prediction_probs_layer')(x)
    return Model(inputs, x, name=f"{backbone}_Classification")

def add_other_class(model):
    # get the dropout layer from the above model
    branch_other = model.get_layer('dropout_layer')\
    # add prediction other or class layer
    branch_other_prediction = layers.Dense(1,name='other_prediction_logit_layer')(branch_other)
    # other probability layer
    branch_other_probs = layers.Activation('sigmoid',name='other_probs_layer')(branch_other_prediction)
    
    # get classes layer from the above model
    branch_classes = model.get_layer('prediction_logits_layer')
    # add other probability to classes layer
    branch_classes_prediction = layers.Multiply()([branch_classes,branch_other_probs])
    # add sigmod activation layer
    branch_classes_probs = layers.Activation('sigmoid',name='classes_probs_layer')(branch_classes_prediction)
    
    return Model(model.input, [branch_classes_probs,branch_other_probs], name=f"{model.name}_finetune")


if __name__ == "__main__":
    model = build_classification_model(num_classes=2)
    model.summary()