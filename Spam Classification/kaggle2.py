import numpy as np
from scipy import io
from sklearn.svm import LinearSVC

data = io.loadmat("./spam_data.mat")
emails = data['training_data']
labels = np.ravel(data["training_labels"])
testData = data['test_data']

#print emails.shape, labels.shape, testData.shape


clf = LinearSVC(C = 1)

clf.fit(emails, labels)
predictions = clf.predict(testData)
#np.savetxt(test.out, predictions,newline='\n')

print "Id,Category"
for i in range(5857):
    category = predictions[i]
    print str(i+1) + "," + str(category)