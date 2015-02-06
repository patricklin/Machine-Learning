import numpy as np
from scipy import io
from sklearn.svm import LinearSVC

data = io.loadmat("./train.mat")
labels = np.ravel(data["train_labels"])
imageData = np.transpose(data["train_images"].reshape(784,60000))

test = io.loadmat("./test.mat")
testData = np.transpose(test["test_images"].reshape(784,10000))

clf = LinearSVC(C = 4E-7)

clf.fit(imageData, labels)
predictions = clf.predict(testData)
#np.savetxt(test.out, predictions,newline='\n')

print "Id,Category"
for i in range(10000):
    category = predictions[i]
    print str(i+1) + "," + str(category)