import numpy as np
import matplotlib.pyplot as plt

# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


DATASET_PATH = 'UCI HAR Dataset/'

TRAIN = "train/"
TEST = "test/"


# Load "X" (Load training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt"\
    for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt"\
    for signal in INPUT_SIGNAL_TYPES
]

Xtrain = load_X(X_train_signals_paths)
Xtest = load_X(X_test_signals_paths)


# Load "y" (training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)


#------------------------------------------------------------------------------
#           Feature Normalizition
#------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler


X_train = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1]*Xtrain.shape[2])
X_test  = Xtest.reshape(Xtest.shape[0], Xtest.shape[1]*Xtest.shape[2])

# Normalizing to a gussian shape distribution : mean=0, variance=0
sc = StandardScaler()
sc.fit_transform(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# scale to range[-1,1]
#min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
#X_train = min_max_scaler.fit_transform(X_train_Guss)
#X_test =  min_max_scaler.fit_transform(X_test_Guss)
#------------------------------------------------------------------------------
#           Multi-layer Perceptron classifier
#------------------------------------------------------------------------------
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def accuracy(cof_mat):
    diag_sum = cof_mat.trace()
    sum_of_all = cof_mat.sum()
    return diag_sum/sum_of_all




#------------------------------------------------------------------------------
#           Cross Validation
#------------------------------------------------
HiddenLayerSizes = [5,20,100,(10,5,)]
Acc_layers, Acc_lr = [],[]
 

d = int(len(X_train)/10)
Folds = [X_train[:d],\
         X_train[d:2*d],X_train[2*d:3*d],X_train[3*d:4*d],\
         X_train[4*d:5*d],X_train[5*d:6*d],X_train[6*d:7*d],\
         X_train[7*d:8*d],X_train[8*d:9*d],X_train[9*d:]
         ]

YFolds = [y_train[:d],\
         y_train[d:2*d],y_train[2*d:3*d],y_train[3*d:4*d],\
         y_train[4*d:5*d],y_train[5*d:6*d],y_train[6*d:7*d],\
         y_train[7*d:8*d],y_train[8*d:9*d],y_train[9*d:]
         ]
 
confs = []
for size in HiddenLayerSizes:  
    temp = []  
    for i in np.arange(0,10):
        Xval = Folds[i]
        Yval = YFolds[i]
        Xtr = np.concatenate(([Folds[i-9],Folds[i-8],Folds[i-7],\
                      Folds[i-6],Folds[i-5],Folds[i-4],\
                      Folds[i-3],Folds[i-2],Folds[i-1]]))

        Ytr = np.concatenate(([YFolds[i-9],YFolds[i-8],YFolds[i-7],\
                      YFolds[i-6],YFolds[i-5],YFolds[i-4],\
                      YFolds[i-3],YFolds[i-2],YFolds[i-1]]))

        classifier = MLPClassifier(solver='adam',\
                        hidden_layer_sizes=size,alpha= 0.05,\
                        max_iter=500, activation = 'relu')
        
        classifier.fit(Xtr,Ytr)

        y_pred = classifier.predict(Xval)
        conf = confusion_matrix(y_pred,Yval)
        Acc = 100*accuracy(conf)
        temp.append(Acc)
        confs.append(conf)
    Acc_layers.append([size,np.mean(temp)])

#----------------- Experiment1 
Acc_tanh=[]
Acc_L2 = []
for size in HiddenLayerSizes:  
    temp = []  
    for i in np.arange(0,10):
        Xval = Folds[i]
        Yval = YFolds[i]
        Xtr = np.concatenate(([Folds[i-9],Folds[i-8],Folds[i-7],\
                      Folds[i-6],Folds[i-5],Folds[i-4],\
                      Folds[i-3],Folds[i-2],Folds[i-1]]))

        Ytr = np.concatenate(([YFolds[i-9],YFolds[i-8],YFolds[i-7],\
                      YFolds[i-6],YFolds[i-5],YFolds[i-4],\
                      YFolds[i-3],YFolds[i-2],YFolds[i-1]]))

        classifier = MLPClassifier(solver='adam',\
                        hidden_layer_sizes=size,alpha= 0.05,\
                        max_iter=500, activation = 'tanh')
        
        classifier.fit(Xtr,Ytr)

        y_pred = classifier.predict(Xval)
        conf = confusion_matrix(y_pred,Yval)
        temp.append(100*accuracy(conf))        
    Acc_tanh.append([size,np.mean(temp)])

   
Acc_L2 = []
for size in HiddenLayerSizes:  
    temp = []  
    for i in np.arange(0,10):
        Xval = Folds[i]
        Yval = YFolds[i]
        Xtr = np.concatenate(([Folds[i-9],Folds[i-8],Folds[i-7],\
                      Folds[i-6],Folds[i-5],Folds[i-4],\
                      Folds[i-3],Folds[i-2],Folds[i-1]]))

        Ytr = np.concatenate(([YFolds[i-9],YFolds[i-8],YFolds[i-7],\
                      YFolds[i-6],YFolds[i-5],YFolds[i-4],\
                      YFolds[i-3],YFolds[i-2],YFolds[i-1]]))

        classifier = MLPClassifier(solver='adam',shuffle=True,\
                        hidden_layer_sizes=size,alpha= 0.001,\
                        max_iter=500, activation = 'relu')
        
        classifier.fit(Xtr,Ytr)

        y_pred = classifier.predict(Xval)
        conf = confusion_matrix(y_pred,Yval)
        Acc = 100*accuracy(conf)
        temp.append(Acc)        
    Acc_L2.append([size,np.mean(temp)])


Acc_L2_big = []
for size in HiddenLayerSizes:  
    temp = []  
    for i in np.arange(0,10):
        Xval = Folds[i]
        Yval = YFolds[i]
        Xtr = np.concatenate(([Folds[i-9],Folds[i-8],Folds[i-7],\
                      Folds[i-6],Folds[i-5],Folds[i-4],\
                      Folds[i-3],Folds[i-2],Folds[i-1]]))

        Ytr = np.concatenate(([YFolds[i-9],YFolds[i-8],YFolds[i-7],\
                      YFolds[i-6],YFolds[i-5],YFolds[i-4],\
                      YFolds[i-3],YFolds[i-2],YFolds[i-1]]))

        classifier = MLPClassifier(solver='adam',shuffle=True,\
                        hidden_layer_sizes=size,alpha= 0.1,\
                        max_iter=500, activation = 'relu')
        
        classifier.fit(Xtr,Ytr)

        y_pred = classifier.predict(Xval)
        conf = confusion_matrix(y_pred,Yval)
        Acc = 100*accuracy(conf)
        temp.append(Acc)        
    Acc_L2_big.append([size,np.mean(temp)])
    
    
    
 


#-------Plot  Experiment1      
Xplot= [2,3,4,5]
Yplot= np.array(Acc_layers)[:,1]

plt.plot(Xplot,np.array(Acc_layers)[:,1],'o--b',\
         label='activation= ReLu, L2 penalty= 0.05')

plt.plot(Xplot,np.array(Acc_L2)[:,1],'o--c',\
         label='activation= ReLu, L2 penalty= 0.001')

plt.plot(Xplot,np.array(Acc_L2_big)[:,1],'o--g',\
         label='activation= ReLu, L2 penalty= 0.1')

plt.plot(Xplot,np.array(Acc_tanh)[:,1],'o--k',\
         label='activation= tanh,L2 penalty= 0.05')

plt.grid(True)   
plt.xticks(Xplot,['5','20','100','10:5(two layers)'])
plt.ylim(70,100)
plt.xlim(1,6)
plt.xlabel('Hidden Layer Size',fontsize ='large')
plt.ylabel('Accuracy[%]',fontsize ='large')
plt.legend()
#plt.title('Effect of hidden layer sixe on accuracy (based on 10-times Cross Validation)')
plt.savefig('exp1.pdf')
plt.show()





#---------------- Experiment2 
Learning_rates = [0.0001,0.001,0.01,0.1]
Acc_clr1,Acc_clr2,Acc_clr3  =[],[],[]


for lr in Learning_rates:  
    temp ,temp2,temp3 = [],[],[]  
    for i in np.arange(0,10):
        Xval = Folds[i]
        Yval = YFolds[i]
        Xtr = np.concatenate(([Folds[i-9],Folds[i-8],Folds[i-7],\
                      Folds[i-6],Folds[i-5],Folds[i-4],\
                      Folds[i-3],Folds[i-2],Folds[i-1]]))

        Ytr = np.concatenate(([YFolds[i-9],YFolds[i-8],YFolds[i-7],\
                      YFolds[i-6],YFolds[i-5],YFolds[i-4],\
                      YFolds[i-3],YFolds[i-2],YFolds[i-1]]))

        clr1 = MLPClassifier(solver='adam',learning_rate_init = lr,\
                        hidden_layer_sizes=(50,),alpha= 0.05,\
                        max_iter=500, activation = 'relu') 
        clr2 = MLPClassifier(solver='adam',learning_rate_init = lr,\
                        hidden_layer_sizes=(100,),alpha= 0.05,\
                        max_iter=500, activation = 'relu') 
        clr3 = MLPClassifier(solver='adam',learning_rate_init = lr,\
                        hidden_layer_sizes=(150,),alpha= 0.05,\
                        max_iter=500, activation = 'relu') 
        clr1.fit(Xtr,Ytr)
        y_pred = clr1.predict(Xval)
        conf = confusion_matrix(y_pred,Yval)
        temp.append(100*accuracy(conf))
        
        clr2.fit(Xtr,Ytr)
        y_pred = clr2.predict(Xval)
        conf = confusion_matrix(y_pred,Yval)
        temp2.append(100*accuracy(conf))
        
        clr3.fit(Xtr,Ytr)
        y_pred = clr3.predict(Xval)
        conf = confusion_matrix(y_pred,Yval)
        temp3.append(100*accuracy(conf))
       
    Acc_clr1.append(np.mean(temp))
    Acc_clr2.append(np.mean(temp2))
    Acc_clr3.append(np.mean(temp3))



Xplot = [0,1,2,3]
plt.plot(Xplot, Acc_clr1,'o-.y',
         label='Hidden Layer Size = 50')

plt.plot(Xplot, Acc_clr2,'o-.b',
         label='Hidden Layer Size = 100')

plt.plot(Xplot, Acc_clr3,'o-.',c='#860090',
         label='Hidden Layer Size = 150')


plt.grid(True)   
plt.xticks(Xplot,['1e-4','1e-3','1e-2','1e-1'])
plt.xlabel('Learning Rate',fontsize ='large')
plt.ylabel('Accuracy[%]',fontsize ='large')
plt.legend()
plt.xlim(-0.4,3.4)
plt.ylim(55,100)
plt.savefig('exp3.pdf')
plt.show()





#----------- Test
classifier = MLPClassifier(solver='adam',learning_rate_init=0.001,
                        hidden_layer_sizes=(100,),alpha= 0.05,\
                        max_iter=250, activation = 'relu',warm_start=False)
classifier.fit(X_train,y_train)

y_test_pred = classifier.predict(X_test)
conf = confusion_matrix(y_test_pred,y_test)




sums = np.sum(conf,axis=1)
walk = conf[0][0] /sums[0]
up = conf[1][1] /sums[1]
down = conf[2][2] /sums[2]
sit = conf[3][3] /sums[3]
stand = conf[4][4] /sums[4]
lay = conf[5][5] /sums[5]


for i in range(6):
    print('{:.2f}'.format(100*conf[i][i] /sums[i]))



        
        
        



















