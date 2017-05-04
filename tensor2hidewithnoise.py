'''
Created on 2017-4-18

@author: Administrator
'''
from datetime import date,datetime,timedelta
import os
import numpy as np
import tensorflow as tf

def listmap(o,p):
    return list(map(o,p))

taskname="3_0pm"
cutime="20170428105404"
prepath = "C:\\Users\\zhilonglu\\Desktop\\kdd\\KDDCUP\\tensordir\\"
inpath=prepath+taskname+"\\"
modelpath=prepath+taskname+"\\"+cutime+"\\"

lossplots=[]

outputs=[]
# 
# a = np.arange(0,12,0.5).reshape(4,-1)
# 
# np.savetxt(path+"a.csv",a,fmt="%.8f",delimiter=',')
# 
# b=np.loadtxt(path+"a.csv",delimiter=',')
# 
# print(b)

def decompose(tensor,inputnum,outputnum,trainnum,validnum,prenum):
    print(tensor.shape)
    trainX=tensor[0:trainnum,0:inputnum]
    trainY=tensor[0:trainnum,inputnum:inputnum+outputnum]
    validX=tensor[trainnum:trainnum+validnum,0:inputnum]
    validY=tensor[trainnum:trainnum+validnum,inputnum:inputnum+outputnum]
    preX=tensor[trainnum+validnum:trainnum+validnum+prenum,0:inputnum]
    return (trainX,trainY,validX,validY,preX)

# tensor=np.loadtxt(inpath+"tensor.csv",delimiter=',')
# trainX,trainY,validX,validY,preX=decompose(tensor,12,6,14,6,7)
# print(trainX.shape,trainY.shape,validX.shape,validY.shape,preX.shape)

def onehide(trainX,trainY,hiddennum,times,keep,modelname,cutime,validX=None,validY=None,valid=False):
    inputnum=len(trainX[0])
    outputnum=len(trainY[0])
    losses1=[]
    losses2=[]
    npx=np.array(trainX)
    npy=np.array(trainY)
    print(np.shape(npx))
    print(np.shape(npy))
    if(valid):
        npx_test=np.array(validX)
        npy_test=np.array(validY)
    sess=tf.InteractiveSession()
    x=tf.placeholder(tf.float32, [None,inputnum])
    W0 = tf.Variable(tf.truncated_normal([inputnum,hiddennum[0]],mean=0, stddev=0.1), name="W0"+cutime)
    W1 = tf.Variable(tf.truncated_normal([hiddennum[0],hiddennum[1]],mean=0, stddev=0.1), name="W1"+cutime)
    W2 = tf.Variable(tf.truncated_normal([hiddennum[1],outputnum],mean=10, stddev=0.1), name="W2"+cutime)
    b0= tf.Variable(tf.ones(hiddennum[0]), name="b0"+cutime)
    b1 = tf.Variable(tf.ones(hiddennum[1]), name="b1"+cutime)
    b2 = tf.Variable(tf.ones(outputnum), name="b2"+cutime)
    saver = tf.train.Saver()
    hidden1=tf.nn.elu(tf.add(tf.matmul(x, W0), b0))
    keep_prob=tf.placeholder(tf.float32)
    hidden_drop1=tf.nn.dropout(hidden1,keep_prob)
    hidden2=tf.nn.elu(tf.add(tf.matmul(hidden_drop1, W1), b1))
    hidden_drop2=tf.nn.dropout(hidden2,keep_prob)
    y=tf.add(tf.matmul(hidden_drop2, W2), b2)
    y_=tf.placeholder(tf.float32,[None,outputnum])
    lossfun=tf.reduce_mean(tf.abs(tf.subtract(y_/y,1)))
    step = tf.Variable(0)
    train_step=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossfun)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(times):
        noiseX=npx+2*np.random.random(npx.shape)-1
        noiseY=npy+2*np.random.random(npy.shape)-1
        sess.run(train_step,feed_dict={x:noiseX,y_:noiseY,keep_prob:keep})
        losses1.append(sess.run(lossfun,feed_dict={x:npx,y_:npy,keep_prob:1}))
        if(valid):
            losses2.append(sess.run(lossfun,feed_dict={x:npx_test,y_:npy_test,keep_prob:1}))
    save_path = saver.save(sess, path+modelname+".ckpt")
    lossplots.append(modelname+"nihe"+","+",".join(listmap(str,losses1[-15000:])))
    print(losses1[-1])
    if(valid):
        lossplots.append(modelname+"yanzheng"+","+",".join(listmap(str,losses2[-15000:])))
        print(losses2[-1])
    preY=sess.run(y,feed_dict={x:npx_test,keep_prob:1})
    np.savetxt(path+"preY.csv",preY,fmt="%.8f",delimiter=',')
    return losses2[-1]

def pred(hiddennum,outputnum,modelname,cutime,preX):
    inputnum=len(trainX[0])
    npx=np.array(trainX)
    npy=np.array(trainY)
    print(np.shape(npx))
    print(np.shape(npy))
    sess=tf.InteractiveSession()
    x=tf.placeholder(tf.float32, [None,inputnum])
    W0 = tf.Variable(tf.truncated_normal([inputnum,hiddennum[0]],mean=0, stddev=0.1), name="W0"+cutime)
    W1 = tf.Variable(tf.truncated_normal([hiddennum[0],hiddennum[1]],mean=0, stddev=0.1), name="W1"+cutime)
    W2 = tf.Variable(tf.truncated_normal([hiddennum[1],outputnum],mean=10, stddev=0.1), name="W2"+cutime)
    b0= tf.Variable(tf.ones(hiddennum[0]), name="b0"+cutime)
    b1 = tf.Variable(tf.ones(hiddennum[1]), name="b1"+cutime)
    b2 = tf.Variable(tf.ones(outputnum), name="b2"+cutime)
    saver = tf.train.Saver()
    hidden1=tf.nn.elu(tf.add(tf.matmul(x, W0), b0))
    keep_prob=tf.placeholder(tf.float32)
    hidden_drop1=tf.nn.dropout(hidden1,keep_prob)
    hidden2=tf.nn.elu(tf.add(tf.matmul(hidden_drop1, W1), b1))
    hidden_drop2=tf.nn.dropout(hidden2,keep_prob)
    y=tf.add(tf.matmul(hidden_drop2, W2), b2)
    y_=tf.placeholder(tf.float32,[None,outputnum])
    lossfun=tf.reduce_mean(tf.abs(tf.subtract(y_/y,1)))
    step = tf.Variable(0)
    train_step=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossfun)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, modelpath+modelname+".ckpt")
    test_X=np.array(preX)
    preY=sess.run(y,feed_dict={x:test_X,keep_prob:1})
    np.savetxt(inpath+"preY.csv",preY,fmt="%.8f",delimiter=',')

# trainX=np.loadtxt(inpath+"trainX.csv",delimiter=',')
# trainY=np.loadtxt(inpath+"trainY.csv",delimiter=',')
# validX=np.loadtxt(inpath+"validX.csv",delimiter=',')
# validY=np.loadtxt(inpath+"validY.csv",delimiter=',')
# preX=np.loadtxt(inpath+"preX.csv",delimiter=',')

tensor=np.loadtxt(inpath+"tensor.csv",delimiter=',')
trainX,trainY,validX,validY,preX=decompose(tensor,12,6,14,6,7)
print(trainX.shape,trainY.shape,validX.shape,validY.shape,preX.shape)

# onehide(trainX, trainY, 6, int(1e4), 0.94, taskname)
pred([24,12],6,taskname, cutime, preX)
 
# for i in range(10):
#     cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
#     print(cutime)
#     path=prepath+taskname+"\\"+cutime+"\\"
#     os.makedirs(path)
#     print(path)
#     lossi=onehide(trainX, trainY, [24,12], int(1.5e4), 1, taskname, cutime, validX, validY, True)
#     with open(path +"lossplots"+datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")+".csv","w") as f:
#         f.write("\n".join(lossplots))
# #     if(lossi<0.09):
# #         break