import matplotlib.pyplot as plt
import numpy as np
from statistics import *

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage import transform
from sklearn import tree


#Model hyperparameters
gamma_list= [0.01, 0.005,0.001,0.0005,0.0001]
c_list=[0.1,0.2,0.5,0.7,1,2,5,7,10]

h_param_comb=[{'gamma':g,'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb)== len(gamma_list)*len(c_list)

#different splits        
train_frac=[0.8,0.7,0.6,0.65,0.5]
test_frac=[0.1,0.2,0.2,0.2,0.25]
dev_frac=[0.1,0.1,0.2,0.15,0.25]
best_test_acc_svm=[]
best_test_acc_DT=[]

#SVM
print("\nSVM:\n")
for i in range(5):

    dev_acc_array_svm=[]
    train_acc_array_svm=[]
    test_acc_array_svm=[]

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
    #label = dataset.target

    #PART:define train/dev/test splits of experiments protocol
    dev_test_frac=1-train_frac[i]
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(data, digits.target, test_size=dev_test_frac, shuffle=True)


    X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test,y_dev_test, test_size=(dev_frac[i])/dev_test_frac, shuffle=True)


    # Create a classifier: a support vector classifier
    best_acc_svm=-1.0
    best_model_svm=None
    best_h_params_svm=None

    print("For split (train,test,val):",train_frac[i],test_frac[i],dev_frac[i])
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
        predicted_dev_svm = clf.predict(X_dev)
        predicted_train_svm = clf.predict(X_train)
        predicted_test_svm = clf.predict(X_test)
        
        cur_acc_svm=metrics.accuracy_score(y_pred=predicted_dev_svm, y_true=y_dev)
        train_acc_svm=metrics.accuracy_score(y_pred=predicted_train_svm, y_true=y_train)
        test_acc_svm=metrics.accuracy_score(y_pred=predicted_test_svm, y_true=y_test)
        
        dev_acc_array_svm.append(cur_acc_svm)
        train_acc_array_svm.append(train_acc_svm)
        test_acc_array_svm.append(test_acc_svm)
        

        #print("Parameters:",cur_h_params)
        print('{:5s} {:.4f} {:5s} {:.1f} {:5s} {:10f} {:15f} {:15f}'.format("Prameters:{ 'Gamma':",cur_h_params["gamma"],", 'C':",cur_h_params["C"],"}",train_acc_svm,cur_acc_svm,test_acc_svm))
            
        if cur_acc_svm>best_acc_svm:
            best_acc_svm=cur_acc_svm
            best_model_svm=clf
            best_h_params_svm=cur_h_params
            #print("Found new best acc with: "+str(cur_h_params))
            #print("New best val accuracy: "+str(cur_acc))

        #Get test set predictions
        # Predict the value of the digit on the test subset
        
    predicted_dev_svm = clf.predict(X_dev)
    predicted_train_svm = clf.predict(X_train)
    predicted_test_svm = clf.predict(X_test)

    cur_acc_svm=metrics.accuracy_score(y_pred=predicted_dev_svm, y_true=y_dev)
    train_acc_svm=metrics.accuracy_score(y_pred=predicted_train_svm, y_true=y_train)
    test_acc_svm=metrics.accuracy_score(y_pred=predicted_test_svm, y_true=y_test)

    #PART: Sanity check of predictions
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted_test_svm):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
        
    #PART: compute evaluation matrics

    #print(cur_h_params)
    print(
        f"\nClassification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted_test_svm)}\n"
    )
        
    #PART: Compute evaluation metrics
    print("Best hyperparameters were:")
    print(cur_h_params)
    print('{:20s} {:15s} {:1s} '.format("    Train accuracy","Val accuracy","Test accuracy"))
    print('{:15f} {:15f} {:15f} '.format(train_acc_svm,cur_acc_svm,test_acc_svm))
    best_test_acc_svm.append(test_acc_svm)

print('\n{:20s} {:15s} '.format("    test mean","test standard deviation"))
print('{:15f} {:15f} '.format(np.mean(best_test_acc_svm),stdev(best_test_acc_svm)))


#Decision Tree
print("\nDecision Tree:\n")
for j in range(5):
    dev_acc_array_DT=[]
    train_acc_array_DT=[]
    test_acc_array_DT=[]

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
    #label = dataset.target

    #PART:define train/dev/test splits of experiments protocol
    dev_test_frac=1-train_frac[j]
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(data, digits.target, test_size=dev_test_frac, shuffle=True)


    X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test,y_dev_test, test_size=(dev_frac[j])/dev_test_frac, shuffle=True)


    # Create a classifier: a support vector classifier
    best_acc_DT=-1.0
    best_model_DT=None
    best_h_params_DT=None

    print("For split (train,test,val):",train_frac[j],test_frac[j],dev_frac[j])
    print('{:43s} {:15s} {:15s} {:10s}'.format(" ","Train accuracy","Val accuracy","Test accuracy"))
    for cur_h_params in h_param_comb:
        clf2 = tree.DecisionTreeClassifier()
        

        #PART setting up hyperparameter
        hyper_params=cur_h_params


        #train model
        # Learn the digits on the train subset
        clf2.fit(X_train, y_train)
        
        
        #print(cur_h_params)
        #Get test set predictions
        predicted_dev_DT = clf2.predict(X_dev)
        predicted_train_DT = clf2.predict(X_train)
        predicted_test_DT = clf2.predict(X_test)
        
        cur_acc_DT=metrics.accuracy_score(y_pred=predicted_dev_DT, y_true=y_dev)
        train_acc_DT=metrics.accuracy_score(y_pred=predicted_train_DT, y_true=y_train)
        test_acc_DT=metrics.accuracy_score(y_pred=predicted_test_DT, y_true=y_test)
        
        dev_acc_array_DT.append(cur_acc_DT)
        train_acc_array_DT.append(train_acc_DT)
        test_acc_array_DT.append(test_acc_DT)
        

        #print("Parameters:",cur_h_params)
        print('{:5s} {:.4f} {:5s} {:.1f} {:5s} {:10f} {:15f} {:15f}'.format("Prameters:{ 'Gamma':",cur_h_params["gamma"],", 'C':",cur_h_params["C"],"}",train_acc_DT,cur_acc_DT,test_acc_DT))
            
        if cur_acc_DT>best_acc_DT:
            best_acc_DT=cur_acc_DT
            best_model_DT=clf2
            best_h_params_DT=cur_h_params
            #print("Found new best acc with: "+str(cur_h_params))
            #print("New best val accuracy: "+str(cur_acc))

        #Get test set predictions
        # Predict the value of the digit on the test subset
        
    predicted_dev_DT = clf2.predict(X_dev)
    predicted_train_DT = clf2.predict(X_train)
    predicted_test_DT = clf2.predict(X_test)

    cur_acc_DT=metrics.accuracy_score(y_pred=predicted_dev_DT, y_true=y_dev)
    train_acc_DT=metrics.accuracy_score(y_pred=predicted_train_DT, y_true=y_train)
    test_acc_DT=metrics.accuracy_score(y_pred=predicted_test_DT, y_true=y_test)

    #PART: Sanity check of predictions
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted_test_DT):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
        
    #PART: compute evaluation matrics

    #print(cur_h_params)
    print(
        f"\nClassification report for classifier {clf2}:\n"
        f"{metrics.classification_report(y_test, predicted_test_DT)}\n"
    )
        
    #PART: Compute evaluation metrics
    print("Best hyperparameters were:")
    print(cur_h_params)
    print('{:20s} {:15s} {:1s} '.format("    Train accuracy","Val accuracy","Test accuracy"))
    print('{:15f} {:15f} {:15f} '.format(train_acc_DT,cur_acc_DT,test_acc_DT))

    best_test_acc_DT.append(test_acc_DT)

print('\n{:20s} {:15s} '.format("    test mean","test standard deviation"))
print('{:15f} {:15f} '.format(mean(best_test_acc_DT),stdev(best_test_acc_DT)))
