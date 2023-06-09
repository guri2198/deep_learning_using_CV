from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from nn.conv.shallownet import ShallowNet

from keras.optimizers import SGD
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import  numpy as np


print("[INFO] loading CIFAR-10 data...")

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
.
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]


print("[INFO] compiling model...")
opt = SGD(lr=0.001)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


print("[INFO] training network...")

Hypo = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=50, verbose=1)


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), Hypo.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), Hypo.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), Hypo.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), Hypo.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./output/shallownet_cifar10_lr0.001.png")
plt.show()