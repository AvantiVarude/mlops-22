import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn import svm
from sklearn import datasets, svm, metrics
from skimage import transform
from sklearn import tree

def svm_prog(i,h_param_comb,train_frac, test_frac,dev_frac):

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
    return y_test,predicted_test_svm,train_acc_svm,cur_acc_svm,test_acc_svm,train_acc_array_svm,train_acc_array_svm,
    dev_acc_array_svm,dev_acc_array_svm,test_acc_array_svm,test_acc_array_svm