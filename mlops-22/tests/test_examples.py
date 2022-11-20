
#import dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split

def test_same_split():
    # TEST case for random state same split
    # train data and test data split
    train_frac = 0.7
    test_frac = 0.15
    dev_frac = 0.15
    
     # actual data and preprocessing
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    #Flatten the image 
    data = digits.images.reshape((n_samples, -1))
    #Get labels from dataset
    label = digits.target
    
    #First splitting: train test split using sklearn train test split and shuffle true and random state 8
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        data, label, test_size=test_frac, shuffle=True, random_state = 8
    )
    #Second splitting: train test split using sklearn train test split and shuffle true and random state 8
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        data, label, test_size=test_frac, shuffle=True, random_state = 8
    )
    
    # checking all of the poits in a train test data are same or not 
    #if same then test case: pass
    #for train x
    assert (X_train1 == X_train2).all()
    #For test x
    assert (X_test1 == X_test2).all()
    #For train y
    assert (y_train1 == y_train2).all()
    #For test y
    assert (y_test1 == y_test2).all()
    


def test_different_split():
    # TEST case for random state same split
    # train data and test data split
    train_frac = 0.7
    test_frac = 0.15
    dev_frac = 0.15
    
    # actual data and preprocessing
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    #flatten the image
    data = digits.images.reshape((n_samples, -1))
    #Get labels from dataset
    label = digits.target
    
    #First splitting: train test split using sklearn train test split and shuffle true
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        data, label, test_size=test_frac, shuffle=True
    )
    #Second splitting: train test split using sklearn train test split and shuffle true
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        data, label, test_size=test_frac, shuffle=True
    )
    
    # checking if the split is different or same
    #For train x
    assert (X_train1 == X_train2).all() == False
    #For test X
    assert (X_test1 == X_test2).all() == False
    #For train y
    assert (y_train1 == y_train2).all() == False
    #for test y
    assert (y_test1 == y_test2).all() == False