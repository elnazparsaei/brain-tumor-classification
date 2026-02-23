from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def build_model(image_shape, num_classes):
    model = Sequential([
        Input(shape=image_shape),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(3,3)),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(3,3)),
        Conv2D(128, (4, 4), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, (4,4), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model