import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage import transform


#Model hyperparameters
gamma_list= [0.01, 0.005,0.001,0.0005,0.0001]
c_list=[0.1,0.2,0.5,0.7,1,2,5,7,10]

h_param_comb=[{'gamma':g,'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb)== len(gamma_list)*len(c_list)


train_frac=0.8
test_frac=0.1
dev_frac=0.1

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


print(f"\nImage size of this dataset is {digits.images[0].shape}")
    
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#PART:define train/dev/test splits of experiments protocol
# Split data into 50% train and 50% test subsets
#8.:10:10 train:dev:test
images = {}
def img_resize(size=8):
  print(f"\n ----x----x----x----x----FOR IMAGE SIZE ({size},{size})----x----x----x----x----\n")
  images[str(size)] = np.zeros((n_samples,size, size))
  for i in range(0,n_samples):
    images[str(size)][i] = transform.resize(digits.images[i],(size, size),anti_aliasing=True)
  data = images[str(size)].reshape((n_samples, -1))
  prediction(data,size)
	
def prediction(data,size):
	dev_test_frac=1-train_frac
	X_train, X_dev_test, y_train, y_dev_test = train_test_split(
	    data, digits.target, test_size=dev_test_frac, shuffle=True
	)


	X_test, X_dev, y_test, y_dev = train_test_split(
	    X_dev_test,y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True)


	# Create a classifier: a support vector classifier
	best_acc=-1.0
	best_model=None
	best_h_params=None

	print('{:43s} {:15s} {:15s} {:10s}'.format(" ","Train accuracy","Val accuracy","Test accuracy"))
	for cur_h_params in h_param_comb:
		clf = svm.SVC()

		#PART setting up hyperparameter
		hyper_params=cur_h_params
		clf.set_params(**hyper_params)


		#train model
		# Learn the digits on the train subset
		clf.fit(X_train, y_train)
		
		
		#print(cur_h_params)
		#Get test set predictions
		predicted_dev = clf.predict(X_dev)
		predicted_train = clf.predict(X_train)
		predicted_test = clf.predict(X_test)
		
		cur_acc=metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
		train_acc=metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
		test_acc=metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
		
		#print("Parameters:",cur_h_params)
		print('{:5s} {:.4f} {:5s} {:.1f} {:5s} {:10f} {:15f} {:15f}'.format("Prameters:{ 'Gamma':",cur_h_params["gamma"],", 'C':",cur_h_params["C"],"}",train_acc,cur_acc,test_acc))
		
		if cur_acc>best_acc:
			best_acc=cur_acc
			best_model=clf
			best_h_params=cur_h_params
			#print("Found new best acc with: "+str(cur_h_params))
			#print("New best val accuracy: "+str(cur_acc))

	#Get test set predictions
	# Predict the value of the digit on the test subset
	
	predicted_dev = clf.predict(X_dev)
	predicted_train = clf.predict(X_train)
	predicted_test = clf.predict(X_test)
	
	cur_acc=metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
	train_acc=metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
	test_acc=metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

	#PART: Sanity check of predictions
	_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
	for ax, image, prediction in zip(axes, X_test, predicted_test):
	    ax.set_axis_off()
	    image = image.reshape(size, size)
	    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
	    ax.set_title(f"Prediction: {prediction}")
	    
	#PART: compute evaluation matrics

	#print(cur_h_params)
	print(
	    f"\nClassification report for classifier {clf}:\n"
	    f"{metrics.classification_report(y_test, predicted_test)}\n"
	)
		
	#PART: Compute evaluation metrics
	print("Best hyperparameters were:")
	print(cur_h_params)
	print('{:20s} {:15s} {:1s} '.format("    Train accuracy","Val accuracy","Test accuracy"))
	print('{:15f} {:15f} {:15f} '.format(train_acc,cur_acc,test_acc))


img_resize() #default size =8
img_resize(4) #image size 4x4
img_resize(10) #image size 10x10
img_resize(16) #image size 16x16
