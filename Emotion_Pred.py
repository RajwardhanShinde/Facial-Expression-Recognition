import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import argparse
import numpy as np
import gc
gc.collect()
#-----------

#Pass image at runtime through terminal
msg = "" 
ap = argparse.ArgumentParser()
ap.add_argument("-image", "--image", required=True)
args = vars(ap.parse_args())

image = load_img(args["image"], target_size=(48,48), grayscale=True)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image /= 255.

model = load_model('model.h5')
pred = model.predict(image)

orig = cv2.imread(args["image"]) 
pred = pred.tolist()
pred = pred[0]
print(pred)
emotion = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#Find the maximum probablity of a image and select that class
lar = 0
for i, ele in enumerate(pred):
    if ele > lar:
        msg = emotion[i]
        lar = ele 
    else:
        pass

#Display image along with class_name
resize = cv2.resize(orig, (800,400))
cv2.putText(resize, msg, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.imshow("Expression", resize)
#press q to exit frame
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
