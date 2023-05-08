from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d","--datasets",required=True, help="path to input dataset")
ap.add_argument("-k","--neighbors",type=int, default=1,help="# of nearest neighbors for classification")
ap.add_argument("-j","--jobs",type=int,default=-1,help="# of jobs for K-NN distance(-1 uses all availabe cores )")
args = vars(ap.parse_args())

print("[info] loading the images ")
imagepaths = list(paths.list_images(args["datasets"]))
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp]) 
(data,labels) = sdl.load(imagepaths,verbose =500)
data = data.reshape((data.shape[0],3072))
print("[info] features matrix : {:.1f}MB".format(data.nbytes/1024*1000.0))
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX , testX , trainY , testY) = train_test_split(data , labels , test_size=0.25 , random_state=42)
print("[info] evaluating KNN classifier")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX),target_names=le.classes_))

