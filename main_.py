from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
import os
import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# report = classification_report(y_true, y_pred)
# classification_report_csv(report)

def fine_tune_test(conv_base):
    conv_base.trainable = True
    layer_4th = conv_base.get_layer('block4_pool').output
    gap = GlobalAveragePooling2D()(layer_4th)
    dropout = Dropout(0.5)(gap)
    dense = Dense(256, activation='relu')(dropout)
    pred = Dense(NUM_CLASSES, activation='softmax')(dense)
    model = Model(inputs=conv_base.inputs, outputs=pred)
    return model


x = 150  # nasnetlarge is problematic because of input size
IMAGE_SIZE = (x, x)
input_ = (x, x, 3)
BATCH_SIZE = 16  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001

if __name__ == '__main__':
    conv_base_ = VGG16(weights='imagenet', include_top=False, input_shape=input_)

    # perform the transfer learning
    conv_base = Model(inputs=conv_base_.inputs, outputs=conv_base_.output)
    # #
    # print(conv_base_.summary())

    data = "augmented"
    dl = "proposed"
    f = 0
    root_path = "//ad.monash.edu/home/User066/csit0004/Desktop/Monkeypox-dataset-2022/Monkeypox-dataset/Augmented_/f" + str(
        f + 1) + '/'
    train_dir = root_path + 'train'
    test_dir = root_path + 'val'
    data_list = os.listdir(train_dir)
    # data_list = os.listdir('D:/COVID/four_classes/splits/f4/train')
    # Delete some classes that may interfere
    print(len(data_list))
    NUM_CLASSES = len(data_list)

    train_s_time = time.clock()
    model = fine_tune_test(conv_base)
    # print(model.summary())
    # model=conv_base_

    # Train datagen here is a preproces
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=50,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.25,
                                       zoom_range=0.1,
                                       channel_shift_range=20,
                                       fill_mode='constant',
                                       # validation_split=0.2
                                       )

    # For multiclass use categorical n for binary us
    train_batches = train_datagen.flow_from_directory(train_dir,
                                                      target_size=IMAGE_SIZE,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      seed=42,
                                                      # subset='training',
                                                      class_mode="categorical",
                                                      # For multiclass use categorical n for binary use binary
                                                      )

    test_datagen = ImageDataGenerator(rescale=1. / 255)  # ,preprocessing_function=CLAHE)

    valid_batches = test_datagen.flow_from_directory(test_dir,
                                                     target_size=IMAGE_SIZE,
                                                     shuffle=True,
                                                     batch_size=BATCH_SIZE,
                                                     seed=42,
                                                     # subset='validation',
                                                     class_mode="categorical")

    # FIT MODEL
    print(len(train_batches))
    print(len(valid_batches))

    print(valid_batches.classes)
    print(valid_batches.class_indices)
    exit(0)
    STEP_SIZE_TRAIN = train_batches.n // train_batches.batch_size
    STEP_SIZE_VALID = valid_batches.n // valid_batches.batch_size

    model.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
                  # optimizer=optimizers.SGD(lr=LEARNING_RATE,momentum=0.9),
                  optimizer=Adam(lr=LEARNING_RATE),
                  #  optimizer=optimizers.Adam(lr_schedule),
                  metrics=['acc'])

    lr_decay = LearningRateScheduler(schedule=lambda epoch: LEARNING_RATE * (0.9 ** epoch))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    callback_list = [es, lr_decay]
    result = model.fit_generator(train_batches,
                                 steps_per_epoch=STEP_SIZE_TRAIN,
                                 validation_data=valid_batches,
                                 validation_steps=STEP_SIZE_VALID,
                                 workers=1,
                                 epochs=NUM_EPOCHS,
                                 callbacks=callback_list
                                 )
    print('Training time:' + str(time.clock() - train_s_time) + 'secs.')
    model.save('testgcp_model.h5')
