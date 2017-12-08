import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.spatial import distance
from sklearn.metrics import precision_recall_fscore_support as score


X_mode = [None,None,None,None,None,None,None,None,None,None,None,None,None];

def workClassValue(x):
    return workClassValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "",1).replace(".", ""))


def workClassValueHelper(xString):
    return {
        #workclass column values
        "Private": computeWeightWorkClass(1.0),
        "Self-emp-not-inc": computeWeightWorkClass(2.0),
        "Self-emp-inc": computeWeightWorkClass(3.0),
        "Federal-gov": computeWeightWorkClass(4.0),
        "Local-gov": computeWeightWorkClass(5.0),
        "State-gov": computeWeightWorkClass(6.0),
        "Without-pay": computeWeightWorkClass(7.0),
        "Never-worked": computeWeightWorkClass(8.0)
    }.get(xString, getMode(1))


def computeWeightWorkClass(x):

    return x


def educationValue(x):
    return educationValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "",1).replace(".", ""))


def educationValueHelper(xString):
    x = 2.0
    return {
        #education column values
        "Preschool": computeWeightEdu(1.0),
        "1st-4th": computeWeightEdu(2.0),
        "5th-6th": computeWeightEdu(3.0),
        "7th-8th": computeWeightEdu(4.0),
        "9th": computeWeightEdu(5.0),
        "10th": computeWeightEdu(6.0),
        "11th": computeWeightEdu(7.0),
        "12th": computeWeightEdu(8.0),
        "HS-grad": computeWeightEdu(9.0),
        "Some-college": computeWeightEdu(10.0),
        "Assoc-acdm": computeWeightEdu(11.0),
        "Assoc-voc": computeWeightEdu(12.0),
        "Prof-school": computeWeightEdu(13.0),
        "Bachelors": computeWeightEdu(14.0),
        "Masters": computeWeightEdu(15.0),
        "Doctorate": computeWeightEdu(16.0),
    }.get(xString, getMode(2))


def computeWeightEdu(x):
    return x


def maritalStatusValue(x):
    return maritalStatusValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "", 1).replace(".", ""))


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


# main function
if __name__ == '__main__':
    #set print options for numpy objects
    np.set_printoptions(suppress=True)

    #get mode for each column for training
    X_mode = getModeArray(np.genfromtxt(fname='adult.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue}))
    cancer_data = np.genfromtxt(fname='adult.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue})

    #removing columns 3 and 5
    cancer_data = trimData(cancer_data)

    #get mode for each column for training
    X_mode = getModeArray(np.genfromtxt(fname='adult.test.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue}))
    cancer_data1 = np.genfromtxt(fname='adult.test.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue})

    cancer_data1 = trimData(cancer_data1)

    print(len(cancer_data1))
    print(cancer_data1)
    print(cancer_data1.shape)

    #train
    X = cancer_data[:, :12]
    print(X.shape)
    X1 = cancer_data[:, 12]

    #test
    Y = cancer_data1[:, :12]
    Y1 = cancer_data1[:, 12]

    weak_Fscore = []
    for K in range(25):
        K_value = K+1

        #used to get n neighbors
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')

        #fitting training set with all values plotting in n dimensions
        neigh.fit(X,X1)

        #predicts training income based on nearest neighbor classifier
        y_pred = neigh.predict(Y)

        #print result
        print("Accuracy is ", accuracy_score(Y1,y_pred)*100,"% for K-Value:",K_value)
        precision, recall, fscore, support = score(Y1, y_pred)

        f = (fscore[0] + fscore[1])/2.0

        weak_Fscore.append(f)
        #print('precision: {}'.format(precision))
        #print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore), ' average is:{}'.format(f))
        #print('support: {}'.format(support))
    max_weak = max(weak_Fscore)
    print(max_weak)
    print(support[1]/(support[0]+support[1]))
