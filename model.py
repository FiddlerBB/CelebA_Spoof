from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from config import IMG_SIZE, LR


def face_mask_cod_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    head_model = base_model.output

    head_model = GlobalAveragePooling2D()(head_model)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(512, activation='relu')(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation='softmax')(head_model)

    vgg_model = Model(base_model.inputs, head_model)
    for layer in base_model.layers:
        layer.trainable = True
    # optim = Adam(learning_rate=LR)
    # vgg_model.compile(loss='categorical_crossentropy', optimizer=optim,
    #               metrics='accuracy')

    return vgg_model


if __name__ == '__main__':
    model = face_mask_cod_model()