import cv2
from tensorflow.keras.models import load_model
import numpy as np

model_path = 'models/eye_detection_224_1676889950.1911855.h5'
model = load_model(model_path)
image_path = '496120.png'

def pre_processing(image):
    image = cv2.imread(image)[:,:,::-1]
    img = cv2.resize(image, (224,224))
    img = np.expand_dims(img, axis=0)
    return img

processed = pre_processing(image_path)
result = model.predict(processed)
label = np.argmax(result)
confidence = np.max(result)
print(label, confidence)
