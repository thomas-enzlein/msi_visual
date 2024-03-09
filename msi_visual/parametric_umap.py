from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import joblib
from msi_visual.normalization import spatial_total_ion_count, total_ion_count, median_ion

class UMAPVirtualStain:
    def __init__(self, start_bin=0, end_bin=None, normalization='spatial_tic'):
        self.umap = MSIParametricUMAP(start_bin=start_bin, end_bin=end_bin, normalization='spatial_tic')

    def fit(self, images, keras_fit_kwargs):
        self.encoder = self.umap.fit(images, keras_fit_kwargs)

    def predict(self, img):
        return self.umap.predict(img, self.encoder)

    def save(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        self.encoder.save(Path(output_folder) / "umap.keras")
        print("saved encoder")
        joblib.dump(self.umap, Path(output_folder) / "msi_umap.joblib")

    def load(self, output_folder):
        self.encoder = tf.keras.models.load_model(Path(output_folder) / "umap.keras")
        
        self.umap = joblib.load(Path(output_folder) / "msi_umap.joblib")


class MSIParametricUMAP:
    def __init__(self, n_components=1, n_training_epochs=1, n_neighbors=50, start_bin=0, end_bin=None, normalization='spatial_tic'):
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.normalization = normalization
        self.n_components = n_components
        self.n_training_epochs = n_training_epochs
        self.n_neighbors = n_neighbors

    def predict(self, img, encoder):
        norm_funtion = {'tic': total_ion_count, 'median': median_ion, 'spatial_tic': spatial_total_ion_count}[self.normalization]
        vector = norm_funtion(img[:, :, self.start_bin:self.end_bin])
        vector = vector.reshape(-1, vector.shape[-1])
        output = encoder.predict(vector, batch_size=1000)    
        output = output.reshape(img.shape[0], img.shape[1], output.shape[-1])
        return output

    def fit(self, images, keras_fit_kwargs={}):
        norm_funtion = {'tic': total_ion_count, 'median': median_ion, 'spatial_tic': spatial_total_ion_count}[self.normalization]
        vector = np.concatenate([norm_funtion(img[:, :, self.start_bin:self.end_bin]).reshape(img.shape[0]*img.shape[1], -1) for img in images], axis=0)
        vector = vector.reshape((-1, vector.shape[-1]))

        dims = vector.shape[-1] 

        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(dims,)),
            tf.keras.layers.Dense(units=256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=self.n_components)
        ])
        
        umap_model = ParametricUMAP(encoder=encoder, 
                                        verbose=True,
                                        n_components=self.n_components,
                                        n_neighbors=self.n_neighbors,
                                        n_training_epochs=self.n_training_epochs,
                                        loss_report_frequency=100,
                                        dims=(dims,),
                                        run_eagerly=True,
                                        keras_fit_kwargs=keras_fit_kwargs)

        umap_model.fit(vector)
        return encoder