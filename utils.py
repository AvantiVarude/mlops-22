import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


#Model hyperparameters
gamma_list= [0.01, 0.005,0.001,0.0005,0.0001]
c_list=[0.1,0.2,0.5,0.7,1,2,5,7,10]

h_param_comb=[{'gamma':g,'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb)== len(gamma_list)*len(c_list)

def test_check_model_saving():
	model_path='/path'
	

train_frac=0.8
test_frac=0.1
dev_frac=0.1

digits = datasets.load_digits()

def test_check_model_saving():
	model_path='/path'
	
def train_save_model(data, label,train_frac,dev_frac,path,model_path)
{
	
	# Split data into 50% train and 50% test subsets
	X_train, X_test, y_train, y_test = train_test_split(
	    data, digits.target, test_size=0.5, shuffle=False
	)
	# Create a classifier: a support vector classifier
	clf = svm.SVC(gamma=0.001)
	
	matric=matrics.accuracy_score
	
}

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
    
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))




# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
    
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

