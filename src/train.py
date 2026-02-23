import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from src.data import get_data_labels, get_dataset
from src.preprocess import preprocess_train, preprocess_test, encode_labels
from src.model import build_model

USER_PATH = "dataset"
image_dim = (168, 168)
batch_size = 32
num_classes = 4
epochs = 50

train_paths, train_index = get_data_labels(USER_PATH + '/Training', random_state=111)
test_paths, test_index = get_data_labels(USER_PATH + '/Testing', random_state=111)

train_ds = get_dataset(train_paths, train_index, image_dim, n_channels=1, batch_size=batch_size)
test_ds = get_dataset(test_paths, test_index, image_dim, n_channels=1, batch_size=batch_size)

train_ds_preprocessed = train_ds.map(preprocess_train).map(lambda i, l: encode_labels(i, l, num_classes))
test_ds_preprocessed = test_ds.map(preprocess_test).map(lambda i, l: encode_labels(i, l, num_classes))

model = build_model((168, 168, 1), num_classes)
optimizer = Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.9925)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, min_lr=1e-4, patience=4, verbose=1)
model_mc = ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

history = model.fit(train_ds_preprocessed, validation_data=test_ds_preprocessed, epochs=epochs, callbacks=[model_rlr, model_mc])