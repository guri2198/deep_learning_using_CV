from pyimagesearch.nn import neuralnetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print("[info] loading MNIST (sample) datasets....")
digits = datasets.load_digits()
data = digits.data.astype("float")
data= (data-data.min())/ (data.max()-data.min())
print("[info] samples: {} dim: {}.format(data.shape[0],data.shape[1])")

(trainX,testX,trainY,testY)=train_test_split(data,digits.target,test_size=0.25)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
print("[INFO] TRANING NETWORK....")
nn = neuralnetwork([trainX.shape[1],32,16,10])
print("[info ] {}".format(nn))
nn.fit(trainX,trainY,epochs=1000)
print("[info] evaluting network....")
predictions = nn.predict(testX)
predictions = nn.argmax(axis=1)
print(classification_report(testY.argmax(axis=1),predictions))