from sklearn import datasets
from sklearn import svm
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from scipy import stats
import numpy as np
import matplotlib.pyplot as plot


X_mode = [None,None,None,None,None,None,None,None,None,None,None,None,None];


#start of lambda functions
def workClassValue(x):
    return workClassValueHelper( str(x).replace("'", "").replace(" ", "").replace("b", "", 1).replace(".", "") )


def workClassValueHelper(xString):
    return {
        #workclass column values
        "Private": computeWeightWorkClass(6.0),
        "Self-emp-not-inc": computeWeightWorkClass(2.0),
        "Self-emp-inc": computeWeightWorkClass(8.0),
        "Federal-gov": computeWeightWorkClass(4.0),
        "Local-gov": computeWeightWorkClass(4.0),
        "State-gov": computeWeightWorkClass(4.0),
        "Without-pay": computeWeightWorkClass(0.0),
        "Never-worked": computeWeightWorkClass(0.0)
    }.get(xString, getMode(1))


def computeWeightWorkClass(x):

    return x*x-2.0*x


def educationValue(x):
    return educationValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "",1).replace(".", ""))


def educationValueHelper(xString):
    x = 2.0
    return {
        #education column values
        "Preschool": computeWeightEdu(1.0),
        "1st-4th": computeWeightEdu(1.0),
        "5th-6th": computeWeightEdu(3.0),
        "7th-8th": computeWeightEdu(3.0),
        "9th": computeWeightEdu(5.0),
        "10th": computeWeightEdu(5.0),
        "11th": computeWeightEdu(5.0),
        "12th": computeWeightEdu(5.0),
        "HS-grad": computeWeightEdu(7.0),
        "Some-college": computeWeightEdu(8.0),
        "Assoc-acdm": computeWeightEdu(10.0),
        "Assoc-voc": computeWeightEdu(10.0),
        "Prof-school": computeWeightEdu(16.0),
        "Bachelors": computeWeightEdu(14.0),
        "Masters": computeWeightEdu(16.0),
        "Doctorate": computeWeightEdu(18.0),
    }.get(xString, getMode(2))


def computeWeightEdu(x):
    return (x+x)*x -2.0*x


def maritalStatusValue(x):
    return maritalStatusValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "",1).replace(".", ""))


def maritalStatusValueHelper(xString):
    return {
        #marital status column values
        "Never-married": 1.0,
        "Divorced": 2.0,
        "Separated": 3.0,
        "Widowed": 4.0,
        "Married-spouse-absent": 5.0,
        "Married-AF-spouse": 6.0,
        "Married-civ-spouse": 7.0
    }.get(xString, getMode(3))


def occupationValue(x):
    return occupationValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "",1).replace(".", ""))


def occupationValueHelper(xString):
    return {
        #occupation column values
        "Tech-support": 1.0,
        "Craft-repair": 2.0,
        "Other-service": 3.0,
        "Sales": 4.0,
        "Exec-managerial": 5.0,
        "Prof-specialty": 6.0,
        "Handlers-cleaners": 7.0,
        "Machine-op-inspct": 8.0,
        "Adm-clerical": 9.0,
        "Farming-fishing": 10.0,
        "Transport-moving": 11.0,
        "Priv-house-serv": 12.0,
        "Protective-serv": 13.0,
        "Armed-Forces": 14.0
    }.get(xString, getMode(4))


def relationshipValue(x):
    return relationshipValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "",1).replace(".", ""))


def relationshipValueHelper(xString):
    return {
        #relationship column values
        "Wife": 1.0,
        "Own-child": 2.0,
        "Husband": 3.0,
        "Not-in-family": 4.0,
        "Other-relative": 5.0,
        "Unmarried": 6.0
    }.get(xString, getMode(5))


def raceValue(x):
    return raceValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "",1).replace(".", ""))


def raceValueHelper(xString):
    return {
        #race column values
        "White": 1.0,
        "Asian-Pac-Islander": 2.0,
        "Amer-Indian-Eskimo": 3.0,
        "Other": 4.0,
        "Black": 5.0,
    }.get(xString, getMode(6))


def genderValue(x):
    return genderValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "",1).replace(".", ""))


def genderValueHelper(xString):
    return {
        #gender
        "Male": 1.0,
        "Female": 2.0
    }.get(xString, getMode(7))


def countryValue(x):
    return countryValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "",1).replace(".", ""))


def countryValueHelper(xString):
    #print(xString + "1")
    return {
        #native country
        "United-States": 1.0,
        "Cambodia": 2.0,
        "England": 3.0,
        "Puerto-Rico": 4.0,
        "Canada": 5.0,
        "Germany": 6.0,
        "Outlying-US(Guam-USVI-etc)": 7.0,
        "India": 8.0,
        "Japan": 9.0,
        "Greece": 10.0,
        "South": 11.0,
        "China": 12.0,
        "Cuba": 13.0,
        "Iran": 14.0,
        "Honduras": 15.0,
        "Philippines": 16.0,
        "Italy": 17.0,
        "Poland": 18.0,
        "Jamaica": 19.0,
        "Vietnam": 20.0,
        "Mexico": 21.0,
        "Portugal": 22.0,
        "Ireland": 23.0,
        "France": 24.0,
        "Dominican-Republic": 25.0,
        "Laos": 26.0,
        "Ecuador": 27.0,
        "Taiwan": 28.0,
        "Haiti": 29.0,
        "Columbia": 30.0,
        "Hungary": 31.0,
        "Guatemala": 32.0,
        "Nicaragua": 33.0,
        "Scotland": 34.0,
        "Thailand": 35.0,
        "Yugoslavia": 36.0,
        "El-Salvador": 37.0,
        "Trinadad&Tobago": 38.0,
        "Peru": 39.0,
        "Hong": 40.0,
        "Holand-Netherlands": 41.0
    }.get(xString, getMode(11))


def salaryValue(x):
    return salaryValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "", 1).replace(".", ""))


def salaryValueHelper(xString):
    return {
        #salary
        "<=50K": 1.0,
        ">50K": -1.0
    }.get(xString, getMode(12))
#end of lamda functions

def euclidian(x):
    return distance.euclidean((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), x, w=None)


def printValBefore(x):
    print(x)
    return getMode(11)


def getMode(column):
    global X_mode
    x = 0.0
    if X_mode[0] is None:
        return x
    x = X_mode[column]
    x = x
    #print(" new mode val " + str(x))
    return x


def getModeArray(X_data):
    X_data_new = trimData(X_data)
    print(X_data_new.shape)

    Xmode = stats.mode(X_data_new)
    x = Xmode[0][0]
    print(x)

    return x

def trimData(X_data):
    #get columns 1-2
    X_data1 = X_data[:, :2]
    #print(X_data1.shape)

    #get column 4
    X_data2 = np.expand_dims(X_data[:, 3], axis=1)
    #print(X_data2.shape)

    #get columns 6-14
    X_data3 = X_data[:, 5:15]
    #print(X_data3.shape)

    #merge all sub X_data sets
    X_data_new = np.concatenate((np.concatenate((X_data1, X_data2), axis=1), X_data3), axis=1)
    #print(X_data_new.shape)

    return X_data_new



if __name__ == '__main__':
    #set print options for numpy objects
    np.set_printoptions(suppress=True)

    #get mode for each column for training
    X_mode = getModeArray(np.genfromtxt(fname='adult.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue}))

    #get training data
    X_data = np.genfromtxt(fname='adult.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue})

    #removing columns 3 and 5
    X_data = trimData(X_data)

    # create blank data structure
    x_training_euclidian = np.empty(shape=[1], dtype='float')
    x_training_euclidian = np.delete(x_training_euclidian, 0, axis=0)

    # compute euclidian distance for first 11 attributes
    #normalize
    for x in X_data[:, :11]:
        x_training_euclidian = np.append(x_training_euclidian, [euclidian(tuple(x))], axis=0)

    #get 12th attribute
    x_training_country = X_data[:, 11]

    #print ndarray size
    print(x_training_euclidian.shape)
    print(x_training_country.shape)

    #add one more column so i can merge later
    x_training_country = np.expand_dims(x_training_country, axis=1)
    x_training_euclidian = np.expand_dims(x_training_euclidian, axis=1)

    #print ndarray size
    print(x_training_euclidian.shape)
    print(x_training_country.shape)

    #merge the 2 ndarrays
    x_training_data = np.concatenate((x_training_country, x_training_euclidian), axis=1)

    #get classifier
    x_training_result = X_data[:, 12]

    #get mode for each column for training
    X_mode = getModeArray(np.genfromtxt(fname='adult.test.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue}))

    #get testing data
    Y_data = np.genfromtxt(fname='adult.test.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue})

    #removing columns 3 and 5
    Y_data = trimData(Y_data)

    # create blank data structure
    y_testing_euclidian = np.empty(shape=[1], dtype='float')
    y_testing_euclidian = np.delete(y_testing_euclidian, 0, axis=0)

    # compute euclidian distance for first 11 attributes
    for y in Y_data[:, :11]:
        y_testing_euclidian = np.append(y_testing_euclidian, [euclidian(tuple(y))], axis=0)

    #get 12th attribute
    y_testing_country = Y_data[:, 11]

    #print ndarray size
    print(y_testing_euclidian.shape)
    print(y_testing_country.shape)

    #add one more column so i can merge later
    y_testing_country = np.expand_dims(y_testing_country, axis=1)
    y_testing_euclidian = np.expand_dims(y_testing_euclidian, axis=1)

    #print ndarray size
    print(y_testing_euclidian.shape)
    print(y_testing_country.shape)

    #merge the 2 ndarrays
    y_testing_data = np.concatenate((y_testing_country, y_testing_euclidian), axis=1)

    #get testing classifier
    y_testing_result = Y_data[:, 12]

    #svm function
    #svc = svm.LinearSVC(C=1.0).fit(x_training_data,x_training_result)
    svc = svm.SVC(kernel='rbf', gamma=1e-2, C=1.0)
    svc.fit(x_training_data, x_training_result)

    #w = svc.coef_
    #predict using testing
    Z = svc.predict(y_testing_data)

    print(Z.shape)
    print(accuracy_score(y_testing_result, Z))
    precision, recall, fscore, support = score(y_testing_result, Z)

    f = (fscore[0] + fscore[1])/2.0

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore), ' average is:{}'.format(f))
    print('support: {}'.format(support))

    print(support[1]/(support[0]+support[1]))

    plot.figure('rbf')
    plot.clf()


    #plot here
    plot.scatter(y_testing_data[:,0], y_testing_data[:, 1], c=y_testing_result,zorder=10, cmap=plot.cm.Paired,edgecolor='k', s=20)

    plot.axis('tight')
    #get min and max for all axis
    x_min = x_training_data[:, 0].min()
    x_max = x_training_data[:, 0].max()
    y_min = x_training_data[:, 1].min()
    y_max = x_training_data[:, 1].max()

    #python meshgrid
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    #get decision function from svm
    Z1 = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])

    #reshape
    Z1 = Z1.reshape(XX.shape)

    #plot color
    plot.pcolormesh(XX, YY, Z1 > 0, cmap=plot.cm.Paired)

    #plot decision boundaries
    plot.contour(XX, YY, Z1, colors=['k', 'k', 'k'],linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plot.show()

