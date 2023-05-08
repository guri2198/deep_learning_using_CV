from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(X):
    return 1.0 / (1+np.exp(-X))
def sigmoid_deriv(X):
    return X * (1 - X)
def predict(X,W):
    Preds = sigmoid_activation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[Preds>0] = 1
    return preds
ap = argparse.ArgumentParser()
ap.add_argument("-e","--epochs",type=float,default=100,help="#of epochs")
ap.add_argument("-a","--alpha",type=float,default=0.01,help="learning rate")
args = vars(ap.parse_args())
(X,Y) = make_blobs(n_samples=1000,n_features=2,centers=2,cluster_std=1.5,random_state=1)
Y =Y.reshape((Y.shape[0],1))
X=np.c_[X,np.ones((X.shape[0]))]
(trainX,testX,trainY,testY)= train_test_split(X,Y,test_size=0.5,random_state=42)
print("[info] training....")
w=np.random.randn(X.shape[1],1)
losses=[]
for epoch in np.arange(0,args["epochs"]):
    preds = sigmoid_activation(trainX.dot(w))

    error =preds - trainY
    loss=np.sum(error**2)
    losses.append(loss)

    d=error*sigmoid_deriv(preds)
    gradient = trainX.T.dot(d)
    w +=-args["alpha"]*gradient
if epoch==0 or (epoch +1)%5 ==0:
    print("[info epoch{} ,loss={:.7f}]".format(int(epoch + 1),loss))
    print("[info evaluting ....]")
    preds = predict(testX,w)
    print(classification_report(testY,preds))
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,))
plt.title("data")
plt.scatter(testX[:,0],testX[:,1],marker="o",c =testY[:,0],s=30)
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]) ,losses)
plt.title("traning loss")
plt.xlabel("Epoch #")
plt.ylabel("loss")
plt.show()
