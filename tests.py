import unittest
import numpy as np
import nibabel as nib
import time
import os
import tensorflow as tf
import nilearn.plotting as plotting
from utils_vif import *
from model_vif import *

class TestDataGenerator(unittest.TestCase):
    
    def setUp(self):
        DATASET_DIR = "/home/mrispec/AUTOAIF_DATA/loos_model"
        self.paths = ['203421_1st_timepoint.nii.gz', '500101_1st_timepoint.nii.gz', '1101970_1st_timepoint.nii.gz']
        # self.paths = ['sub-1100841_ses-02_desc-hmc_DCE.nii.gz']
        self.directory = '/home/mrispec/AUTOAIF_DATA/loos_model/USC/'
        self.batch_size = 1
        self.input_size = (256, 256, 32, 32)
        self.shuffle = True
        self.data_augmentation = True
        # self.currentgenerator = DataGenerator(self.paths, self.directory, self.batch_size, self.input_size, self.shuffle, self.data_augmentation)
        # train_set = load_data(os.path.join(self.directory, 'images/', self.paths[0]))
        # self.tfgenerator = tf.data.Dataset.from_generator(lambda: train_generator(os.path.join(DATASET_DIR,"train/"), True, True, self.paths), output_types=(tf.float32, (tf.float32, tf.float32, tf.float32))).batch(256).prefetch(tf.data.AUTOTUNE)
        # self.dataset = tf.data.Dataset.from_generator(generator=self.generator,
        #                                      output_types=(tf.float32, tf.float32),
        #                                      output_shapes=((self.batch_size, *self.input_size), (self.batch_size, 1)))
        self.num_epochs = 2
        
    # def test_len(self):
    #     self.assertEqual(len(self.currentgenerator), len(self.paths) // self.batch_size)
        
    # def test_getitem(self):
    #     batch_images, (batch_cof, batch_curve, batch_vol, batch_quality) = self.currentgenerator[0]
    #     self.assertEqual(batch_images.shape, (self.batch_size, *self.input_size))
    #     # self.assertEqual(batch_cof.shape, (self.batch_size, 3))
    #     self.assertEqual(batch_curve.shape, (self.batch_size, self.input_size[3]))
    #     self.assertEqual(batch_vol.shape, (self.batch_size, 1))
    #     self.assertEqual(batch_quality.shape, (self.batch_size, 1))
        
    # # def test_on_epoch_end(self):
    # #     self.generator.on_epoch_end()
    # #     self.assertFalse(np.array_equal(self.generator.indexes, np.arange(len(self.paths))))
        
    # def test_data_generation(self):
    #     list_IDs_temp = self.paths[:self.batch_size]
    #     batch_images, (batch_masks, batch_curve, batch_vol, batch_quality) = self.currentgenerator._DataGenerator__data_generation(list_IDs_temp)
    #     self.assertEqual(batch_images.shape, (self.batch_size, *self.input_size))
    #     self.assertEqual(batch_masks.shape, (self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2], 1))
    #     self.assertEqual(batch_curve.shape, (self.batch_size, self.input_size[3]))
    #     self.assertEqual(batch_vol.shape, (self.batch_size, 1))
    #     self.assertEqual(batch_quality.shape, (self.batch_size, 1))
    #     #print a batch of curves
    #     print(batch_curve)
    #     print(batch_vol)

    def test_TFRecord(self):
        imgs = [os.path.join(self.directory, 'images/', img) for img in self.paths]
        masks = [os.path.join(self.directory, 'masks/', mask) for mask in self.paths]
        print(imgs)
        img = self.paths[0]
        print(img)
        img = os.path.join(self.directory, 'images/', img)
        mask = os.path.join(self.directory, 'masks/', img)
        write_records(imgs, masks, len(self.paths), './test/train')

        list_of_records=['./test/train_000-of-000.tfrecords']
        batch_size=1
        ds = get_batched_dataset(list_of_records, batch_size=batch_size, shuffle_size=1)
        image, label = next(iter(ds))
        _ = plt.imshow(image[0, :, :, 0, 0], cmap='gray')
        plt.show()
        # _ = plt.title(label)
        def visualize(original, augmented):
            fig = plt.figure()
            plt.subplot(1,2,1)
            plt.title('Original image')
            plt.imshow(original[0, :, :, 0, 0], cmap='gray')

            plt.subplot(1,2,2)
            plt.title('Augmented image')
            plt.imshow(augmented[0, :, :, 0, 0], cmap='gray')

        # flipped = tf.image.flip_left_right(image[0, :, :, :, 0])
        # visualize(image, flipped)


        # (Xs, Ys) = next(ds.as_numpy_iterator())

        # (batch_size, )
        # order will depend on shuffle (turn off all shuffling to verify order)
        # print(Ys.shape)

        # (batch_size, x_dim, y_dim, z_dim, 1)
        # print(Xs.shape)
        model = unet3d( img_size        = (X_DIM, Y_DIM, Z_DIM, T_DIM),
                learning_rate   = 1e-3,
                learning_decay  = 1e-9,
                weights         = [0, 1, 0])

        model.fit(ds, validation_data=ds, epochs=1, steps_per_epoch=len(self.paths)//batch_size, validation_steps=len(self.paths)//batch_size)

    # def test_TensorRT(self):
    #     import tensorflow as tf
    #     from tensorflow.python.compiler.tensorrt import trt_convert as trt

    #     list_of_records=['train_000-of-000.tfrecords']
    #     train = get_batched_dataset(list_of_records, batch_size=self.batch_size, shuffle_size=1)
    #     list_of_records=['val_000-of-000.tfrecords']
    #     val = get_batched_dataset(list_of_records, batch_size=self.batch_size, shuffle_size=1)
    #     model = unet3d( img_size        = (X_DIM, Y_DIM, Z_DIM, T_DIM),
    #             learning_rate   = 1e-3,
    #             learning_decay  = 1e-9,
    #             weights         = [0, 1, 0])

    #     model.fit(train, validation_data=val, epochs=1, steps_per_epoch=len(self.paths)//self.batch_size, validation_steps=len(self.paths)//self.batch_size)
    #     model.save('/home/mrispec/AUTOAIF_DATA/weights/trttest/model.keras')

    #     # Load the model
    #     model = tf.keras.models.load_model('/home/mrispec/AUTOAIF_DATA/weights/trttest/model.keras', safe_mode=False)

    #     converter = trt.TrtGraphConverterV2(input_saved_model_dir=model)

    #     converter.convert()

    #     converter.save('/home/mrispec/AUTOAIF_DATA/weights/trttest/model_weight_TRT')

    #     trt_model = tf.keras.models.load_model('/home/mrispec/AUTOAIF_DATA/weights/trttest/model_weight_TRT')


    #     print('Loading data')
    #     volume_img = nib.load('/home/mrispec/AUTOAIF_DATA/loos_model/train/203421_1st_timepoint.nii')
    #     print(volume_img.shape)
    #     volume_data = volume_img.get_fdata()

    #     print('Preprocessing')
    #     vol_pre = preprocessing(volume_data)

    #     trt_model.predict(vol_pre)

        # print('Loading model')
        # model = unet3d(img_size = (X_DIM, Y_DIM, Z_DIM, T_DIM),\
        #                 learning_rate = 1e-3,\
        #                 learning_decay = 1e-9)
        # model.trainable = False
        # model.load_weights(args.model_weight_path)

        # print('Prediction')
        # y_pred_mask, y_pred_vf, _ = model.predict('/home/mrispec/AUTOAIF_DATA/weights/newtftest/model_weight.h5')

        
if __name__ == '__main__':
    unittest.main()