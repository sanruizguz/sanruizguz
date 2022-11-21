# Tensorflow, Pillow, Scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.layers import  Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras import optimizers
import matplotlib.pyplot as plt
import tensorflow
import os
import warnings
import matplotlib as plt
import numpy as np
warnings.filterwarnings('ignore')

import tensorflow as tf
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

run = neptune.init_run(
    project="sanruizguz/Birdsong-CNN",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZTE0N2RiYS02ZGU5LTQzMWItODE1NC1jYWFlM2MwYjdiMDIifQ==",
)  # your credentials

datagen = ImageDataGenerator(featurewise_center=False,featurewise_std_normalization=False)

params = {
    "epochs": 200,      # Number of epochs (times the whole dataset will go through the CNN)
    "dropout": 0,    # Dropput rate of the dropout layers
    "batch_size" : 22, # Number of spectrograms that will go through the CNN together
    "lr": 0.0001,       # Learning rate
    "Height": 224,     # Number of pixels of the spectrogram in the y axis   
    "Width": 224,      # Number of pixels of the spectrogram in the x axis
}
run["parameters"] = params

# Paths of the sets
train_path='J:/SETS/SETS_OXA/train'
val_path='J:/SETS/SETS_OXA/validation'
test_path='J:/SETS/SETS_OXA/test'

#---------------------------------LOAD DATA + ADD THE AUGMENTATION------------------------------------------------------------
# load and iterate training dataset
                                                                                              
train_generator = datagen.flow_from_directory(directory =train_path,
                                              class_mode='categorical', # For multilabel purposes
                                              batch_size= params["batch_size"],
                                              color_mode="rgb", # rgb for three channels
                                              target_size=(params["Height"], params["Width"]),
                                              shuffle=True
)
val_generator = datagen.flow_from_directory(directory =val_path,
                                            class_mode='categorical',
                                            batch_size=params["batch_size"],
                                            color_mode="rgb",
                                            target_size=(params["Height"], params["Width"]),
                                            shuffle=True
)
test_generator = datagen.flow_from_directory(directory =test_path,
                                            class_mode='categorical',
                                            batch_size=params["batch_size"],
                                            color_mode="rgb",
                                            target_size=(params["Height"], params["Width"]),
                                            shuffle=True
)
#---------------------------------------VERIFY SHAPES OF THE DATA--------------------------------------------------------------
# confirm the iterator works
batchX, batchy = train_generator.next()
print('Training set batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
batchX, batchy = val_generator.next()
print('Validation set batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
batchX, batchy = test_generator.next()
print('testing set batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

#-------------------------------------IMPORTING RESNET50----------------------------------------------------------------
input_t = tensorflow.keras.Input(shape=(params["Height"],params["Width"],3))
pretrained_model = ResNet50(weights='imagenet', # Pre-trained network
                input_shape=(params["Height"],params["Width"],3),
                input_tensor=input_t,
                include_top=False)

# Freezing the base model: Do not allow updates to the layers of the pre-trained ANN, just in the layers we add.
for layer in pretrained_model.layers:
#------WITHOUT FINE TUNING THE MODEL-----------------------------------------------------------------------------------
    layer.trainable = False
#------FINE TUNING THE MODEL--------------------------------------------------------------------------------------------
#     layer.trainable = True

#------------------------------------------ NETWORK ARCHITECTURE---------------------------------------------------------
# We need to specify that the last layer of the ResNet (pre-trained ANN) will be the first one of our model
#last_layer = pretrained_model.get_layer('conv5_block3_out')   # Name of the last layer of the pre-trained network
#print(last_layer.output_shape)

model = Sequential()
model.add(pretrained_model)

model.add(Conv2D(64, (10, 10), padding="same", activation="relu"))
model.add(Conv2D(128, (7, 7), padding="same", activation="relu"))
model.add(Conv2D(256, (4, 4), padding="same", activation="relu"))


model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='sigmoid'))
model.summary()


#----------------------------------------COMPILING THE MODEL------------------------------------------------------------
# Using Adam optimizer for better performance, change the learning rate (lr)
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=params["lr"])  # Type of optimizer and learning rate
                                                        

model.compile(optimizer= optimizer,
              loss="categorical_crossentropy",# categorical cross entropy used as a loss function for multi-class
                                             # classification problems where there are two or more output labels.
              metrics= tensorflow.keras.metrics.Accuracy())                       #Metrics
                       #tensorflow.keras.metrics.Precision(),
                       #tensorflow.keras.metrics.Recall()))
neptune_cbk = NeptuneCallback(run=run, base_namespace="training")



#----------------------------------------TRAINING THE MODEL--------------------------------------------------------------
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = val_generator.n // val_generator.batch_size

#Early Stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# fit model
history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=params["epochs"],
                    callbacks=[neptune_cbk]
                    )

eval_metrics = model.evaluate(test_generator, verbose=0)
for j, metric in enumerate(eval_metrics):
    run["eval/{}".format(model.metrics_names[j])] = metric

run.stop()

import matplotlib.pyplot as plt
BASE_OUTPUT = "J:/"                                                                     #Path for the plots
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

def plot_training(H, plotpath):
    # construct a plot that plots and saves the training history
   plt.style.use("ggplot")
   plt.figure()
   plt.plot(H.history["loss"], label="train_loss")
   plt.plot(H.history["val_loss"], label="val_loss")
   plt.plot(H.history["accuracy"], label="train_acc")
   plt.plot(H.history["val_accuracy"], label="val_acc")
   plt.title("Training Accuracy")
   plt.xlabel("Epoch #")
   plt.ylabel("Accuracy")
   plt.legend(loc="lower left")
   plt.savefig(plotpath)

print("[INFO] plotting training history...")
plot_training(history, PLOT_PATH)














