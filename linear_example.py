import numpy as np 
import cv2

labels = [ "dog" , "cat" , "pandas"]
np.random.seed(5)
w=np.random.randn(3,3072)
b=np.random.randn(3)
oriig = cv2.imread("dogs_00021.jpg")
image = cv2.resize(oriig, (32, 32)).flatten()
scores = w.dot(image) + b
for(label,score) in zip(labels,scores):
  print("[info] {}: {:.2f}".format(label,score))
cv2.putText(oriig,"Label: {}".format(label[np.argmax(scores)]),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
cv2.imshow("dogs_00021.jpg",oriig)
cv2.waitKey(0)
