from ultralytics import YOLO
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

model = YOLO('best.pt')

input_dir = ('images')
output_dir = ('output')

os.makedirs(output_dir,exist_ok=True)

img_path = ('Test Pictures/403U73031.jpg')

output_dir = ('output/test')
img = cv2.imread(img_path)

results = model(img)

# # Define the custom metric
def MSE(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model1 = tf.keras.models.load_model('my_model_keras.keras', custom_objects={'MSE': MSE})

for i, result in enumerate(results[0].boxes.xyxy):
    x_min, y_min, x_max, y_max = map(int, result)
    cropped_img = img[y_min:y_max, x_min:x_max]
    cropped_img = cv2.resize(cropped_img,(224,224))
    cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_RGB2GRAY)
    cropped_img = np.expand_dims(cropped_img,axis=0)
    cleaned = model1.predict(cropped_img/255.0)
    plt.imshow(cleaned[0],cmap='gray')
    plt.show()

#
#             # # Save the cropped image to the output directory
#             # output_crop_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_crop_{i}.jpg')
#             # cv2.imwrite(output_crop_path, cropped_img)
#             # print(f"Cropped image saved at: {output_crop_path}")
#
