import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import pickle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


def create_model(input_shape, pool_size):
    """ a fully convolutional neural network for detecting middle line.
        inputs are road images in the shape of 150 x 200 x 3 (BGR) with
        the labels as 150 x 200 x 1 (just one channel with a drawn line).
        model created by Michael Virgo https://github.com/mvirgo/MLND-Capstone
        I changed image size and instead of area between two lines
        this model predicts only central yellow line
    """
    # Create the actual neural network here
    model = Sequential()
    # Normalizes incoming inputs. First layer needs the input shape to work
    model.add(BatchNormalization(input_shape=input_shape))

    # Below layers were re-named for easier reading of model summary; this not necessary
    # Conv Layer 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))
    
    # Conv Layer 2
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

    # Pooling 1
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 3
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    model.add(Dropout(0.2))

    # Conv Layer 4
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    model.add(Dropout(0.2))

    # Conv Layer 5
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    model.add(Dropout(0.2))

    # Pooling 2
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 6
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    model.add(Dropout(0.2))

    # Conv Layer 7
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    model.add(Dropout(0.2))

    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size))

    # Upsample 1
    model.add(UpSampling2D(size=pool_size))

    # Deconv 1
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
    model.add(Dropout(0.2))

    # Deconv 2
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
    model.add(Dropout(0.2))

    # Upsample 2
    model.add(UpSampling2D(size=pool_size))

    # Deconv 3
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
    model.add(Dropout(0.2))

    # Deconv 4
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
    model.add(Dropout(0.2))

    # Deconv 5
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
    model.add(Dropout(0.2))

    # Upsample 3
    model.add(UpSampling2D(size=pool_size))

    # Deconv 6
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))

    return model


def main():
    
    test_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(test_dir_path, 'full_conv_network', 'full_conv_train.p')
    labels_path = os.path.join(test_dir_path, 'full_conv_network', 'full_conv_labels.p')

    # dump images in pickle file
    # import glob
    # import cv2
    # img_path = os.path.join(test_dir_path, 'full_conv_network', 'dataset', 'labels', '*')
    # images_list = glob.glob(img_path)
    # print(images_list)
    # images = []
    # file = open(labels_path,'wb')
    # for file_name in images_list:
    #     im = cv2.imread(file_name)
    #     im = im[1:im.shape[0]-1, 2:im.shape[1]-3]
    #     # images.append(im)
    #     images.append(im[:, :, 2])
    # images = np.array(images)
    # print(images.shape)
    # images = images[:, :, :, None]
    # pickle.dump(images,file)
    # file.close()

    # Load training images
    train_images = pickle.load(open(train_path, "rb" ))

    # # Load image labels
    labels = pickle.load(open(labels_path, "rb" ))

    # # # Make into arrays as the neural network wants these
    train_images = np.array(train_images)
    labels = np.array(labels)

    # Normalize labels - training images get normalized to start in the network
    labels = labels / 255
        
    # # Shuffle images along with their labels, then split into training/validation sets
    train_images, labels = shuffle(train_images, labels)
    # # Test size may be 10% or 20%
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

    # # Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
    batch_size = 128
    epochs = 10
    pool_size = (2, 2)
    input_shape = X_train.shape[1:]
    
    # # Create the neural network
    model = create_model(input_shape, pool_size)

    # # Using a generator to help the model use less data
    # # Channel shifts help with shadows slightly
    datagen = ImageDataGenerator() #channel_shift_range=0.2
    datagen.fit(X_train)
    
    # # Compiling and training the model
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
    epochs=epochs, verbose=1, validation_data=(X_val, y_val))
    
    # # Freeze layers since training is done
    model.trainable = False
    model.compile(optimizer='Adam', loss='mean_squared_error')

    # # Save model architecture and weights
    model_path = os.path.join(test_dir_path, 'full_conv_network', 'FCNN_model2.h5')
    model.save(model_path)

    # # Show summary of model
    model.summary()

if __name__ == '__main__':
    main()