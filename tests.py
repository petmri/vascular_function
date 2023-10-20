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
        self.paths = ['203421_1st_timepoint.nii', '500101_1st_timepoint.nii', '1101970_1st_timepoint.nii']
        self.directory = '/home/mrispec/AUTOAIF_DATA/loos_model/train/'
        self.batch_size = 2
        self.input_size = (256, 256, 32, 32)
        self.shuffle = True
        self.data_augmentation = True
        self.currentgenerator = DataGenerator(self.paths, self.directory, self.batch_size, self.input_size, self.shuffle, self.data_augmentation)
        # train_set = load_data(os.path.join(self.directory, 'images/', self.paths[0]))
        self.tfgenerator = tf.data.Dataset.from_generator(lambda: train_generator(os.path.join(DATASET_DIR,"train/"), True, True, self.paths), output_types=(tf.float32, (tf.float32, tf.float32, tf.float32))).batch(256).prefetch(tf.data.AUTOTUNE)
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

    # def test_benchmark(self):
    #     dataset = self.tfgenerator
    #     start_time = time.perf_counter()
    #     for epoch_num in range(self.num_epochs):
    #         for sample in dataset:
    #             # Performing a training step
    #             time.sleep(0.01)
    #     print("Execution time:", time.perf_counter() - start_time)

    # def test_TFRecord(self):
        # imgs = [os.path.join(self.directory, 'images/', img) for img in self.paths]
        # masks = [os.path.join(self.directory, 'masks/', mask) for mask in self.paths]
        # print(imgs)
        # img = self.paths[0]
        # print(img)
        # img = os.path.join(self.directory, 'images/', img)
        # mask = os.path.join(self.directory, 'masks/', img)
        # write_records(imgs, masks, len(self.paths), 'train')

        # list_of_records=['train_000-of-000.tfrecords']
        # batch_size=1
        # ds = get_batched_dataset(list_of_records, batch_size=batch_size, shuffle_size=1)

        # (Xs, Ys) = next(ds.as_numpy_iterator())

        # (batch_size, )
        # order will depend on shuffle (turn off all shuffling to verify order)
        # print(Ys.shape)

        # (batch_size, x_dim, y_dim, z_dim, 1)
        # print(Xs.shape)
        # model = unet3d( img_size        = (X_DIM, Y_DIM, Z_DIM, T_DIM),
        #         learning_rate   = 1e-3,
        #         learning_decay  = 1e-9,
        #         weights         = [0, 1, 0])

        # model.fit(ds, validation_data=ds, epochs=1, steps_per_epoch=len(self.paths)//batch_size, validation_steps=len(self.paths)//batch_size)

    # def test_TFR_image(self):
    #     img = nib.load(os.path.join(self.directory, 'images/', self.paths[0]))
    #     mask = nib.load(os.path.join(self.directory, 'masks/', self.paths[0]))

    #     def image_example(img_str, mask_str):
    #         image_shape = tf.io.decode_raw(img_str, tf.float32).shape

    #         feature = {
    #             'height': _int64_feature(image_shape[0]),
    #             'width': _int64_feature(image_shape[1]),
    #             'depth': _int64_feature(image_shape[2]),
    #             'label': _int64_feature(mask_str),
    #             'image_raw': _bytes_feature(img_str),
    #         }

    #         return tf.train.Example(features=tf.train.Features(feature=feature))
        
    #     for line in str(image_example(img, mask)).split('\n'):
    #         print(line)

    def test_TensorRT(self):
        import tensorflow as tf
        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        # Load the model
        model = tf.keras.models.load_model('/home/mrispec/AUTOAIF_DATA/weights/newtftest/model_weight.h5')

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=model)

        converter.convert()

        converter.save('/home/mrispec/AUTOAIF_DATA/weights/newtftest/model_weight_TRT')

        trt_model = tf.keras.models.load_model('/home/mrispec/AUTOAIF_DATA/weights/newtftest/model_weight_TRT')


        print('Loading data')
        volume_img = nib.load('/home/mrispec/AUTOAIF_DATA/loos_model/train/203421_1st_timepoint.nii')
        print(volume_img.shape)
        volume_data = volume_img.get_fdata()

        print('Preprocessing')
        vol_pre = preprocessing(volume_data)

        trt_model.predict(vol_pre)

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