import numpy as np
from scipy import io
from sklearn.svm import LinearSVC

data = io.loadmat("./train.mat")
labels = np.ravel(data["train_labels"])
imageData = np.transpose(data["train_images"].reshape(784,60000))


indices = np.random.permutation(imageData.shape[0])
trainInd, testInd = indices[:10000], indices[-10000:]
folds = np.asarray(np.split(trainInd,10))

cvalues = [4E-7, 5E-7, 6E-7]


def test_folds():
	for c in cvalues:
		clf = LinearSVC(C = c)
		scores = np.zeros(10)
		for i in range(10):
			validate = folds[i]
			others = [x for x in range(10) if x != i]
			training = np.concatenate(folds[others])

			clf.fit(imageData[training], labels[training])
			score = clf.score(imageData[validate], labels[validate])
			scores[i] = score
			#print "Accuracy after training on fold number {0} with C value {1} is {2}".format(i, c, score)
		print "Average accuracy for C value {0} is {1}".format(c, np.mean(scores))

def validate():
	for c in cvalues:
		clf = LinearSVC(C = c)
		clf.fit(imageData[trainInd], labels[trainInd])
		validateScore = clf.score(imageData[testInd], labels[testInd])
		print "Accuracy for C value {0} on validation set is is {1}".format(c, validateScore)

test_folds()
validate()