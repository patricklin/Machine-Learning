from scipy import io
from sklearn.svm import LinearSVC 
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

#problem 1: bar plot of error rate for 100, 200, 500, 1,000, 2,000, 5,000, and 10,000 training sets
trainingSize = [100, 200, 500, 1000, 2000, 5000, 10000]
data = io.loadmat("./train.mat")
labels = np.ravel(data["train_labels"])
#print data["train_images"].shape	#28,28,60k
imageData = np.transpose(data["train_images"].reshape(784,60000))
#print imageData.shape		#60000 vectors of 784 ints each ranging from 0-255
errorRate = []

clf = LinearSVC()
for i in trainingSize:
	indices = np.random.permutation(imageData.shape[0])
	trainInd, testInd = indices[:i], indices[-10000:]
	
	clf.fit(imageData[trainInd], labels[trainInd])
	
	score = clf.score(imageData[testInd], labels[testInd])
	errorRate.append(score)
	print "Accuracy after training on size of {0} is {1}".format(i, score)
	
	predictions = clf.predict(imageData[testInd])
	cm = confusion_matrix(labels[testInd],predictions)

	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('Actual Digit')
	plt.xlabel('Predicted Digit')
	#plt.show()

plt.plot(trainingSize, errorRate)
plt.ylabel('Accuracy Rate')
plt.xlabel('Training Set Size')
plt.title('Linear SVM Using Raw Pixels')
plt.show()