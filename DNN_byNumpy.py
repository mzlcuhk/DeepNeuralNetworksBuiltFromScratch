import numpy as np
from numpy.core.umath_tests import inner1d


def Jfunction(W, X, Y):
    Z = np.dot(X,W) 
    expZ = np.exp(Z)
    Y_hat = expZ/expZ.sum(axis=1)[:,None]
    M = X.shape[0]
    #J_ce = -np.trace(np.dot(np.log(Y_hat),Y.T))         #too slow
    #J_ce = -np.sum(inner1d(np.dot(np.log(Y_hat),Y.T)))    #too slow
    J_ce = -np.einsum('ij,ji->',np.log(Y_hat+1e-15),Y.T)   # good
    J_ce = J_ce/M
    return J_ce

def gradJ(W, X, Y):
    Z = np.dot(X,W) 
    expZ = np.exp(Z)
    Y_hat = expZ/expZ.sum(axis=1)[:,None]
    loss = Y_hat - Y
    gradj_ce = np.dot(X.T,loss)
    return gradj_ce

def graddescent(trainX,trainY, testX,testY):
    W = np.zeros([trainX.shape[1],trainY.shape[1]])
    eps =  1e-5
    for i in range(0,3000):
        grad = gradJ(W, trainX, trainY)
        W = W - eps*grad
        if i%100 == 0 :
            J = Jfunction(W,trainX,trainY)
            print 'the',i,'round cost: ',J
    return W

def classid(W,X):
    Z = np.dot(X,W) 
    expZ = np.exp(Z)
    Y_hat = expZ/expZ.sum(axis=1)[:,None]
    classid = np.argmax(Y_hat,axis=1)
    return classid

def accuracy(W, X, Y_true):
    cid_hat = classid(W,X)
    cid_true = np.argmax(Y_true,axis=1)
    result = np.equal(cid_hat,cid_true).mean()
    return result  
    
def myrun(trainX,trainY, testX,testY):
    W = graddescent(trainX,trainY, testX,testY)
    print "Trarning cost: {}".format(Jfunction(W,trainX,trainY))
    print "Testing cost: {}".format(Jfunction(W,testX,testY))
    print "Training accuracy: {}".format(accuracy(W,trainX,trainY))
    print "Testing accuracy: {}".format(accuracy(W,testX,testY))


    
trainingX = np.load("mnist_train_images.npy")#[:25000,:]
trainingY = np.load("mnist_train_labels.npy")#[:25000,:]
testingX = np.load("mnist_test_images.npy")#[:25000,:]
testingY = np.load("mnist_test_labels.npy")#[:25000,:]

print 'start'
print "the shape of trainX :", trainingX.shape
print "the shape of trainY :", trainingY.shape
myrun(trainingX,trainingY,testingX,testingY)

