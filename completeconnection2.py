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

taskname="3_1pm"
cutime="20170503182708"
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

def decompose(tensor,inputnum,outputnum,validnum,prenum):
    print(tensor.shape)
    trainnum=tensor.shape[0]-validnum-prenum
    trainX=tensor[0:trainnum,0:inputnum]
    trainY=tensor[0:trainnum,inputnum:inputnum+outputnum]
    validX=tensor[trainnum:trainnum+validnum,0:inputnum]
    validY=tensor[trainnum:trainnum+validnum,inputnum:inputnum+outputnum]
    preX=tensor[trainnum+validnum:trainnum+validnum+prenum,0:inputnum]
    return (trainX,trainY,validX,validY,preX)

def fcn(trainX,trainY,hiddennum,times,keep,modelname,cutime,validX=None,validY=None):
    inputnum=len(trainX[0])
    outputnum=len(trainY[0])
    losses1=[]
    losses2=[]
    losscol=[]
    npx=np.array(trainX)
    npy=np.array(trainY)
    print(np.shape(npx))
    print(np.shape(npy))
    npx_test=np.array(validX)
    npy_test=np.array(validY)
    nodenums=[inputnum]+hiddennum+[outputnum]
    sess=tf.InteractiveSession()
    x=tf.placeholder(tf.float32, [None,inputnum])
    y_=tf.placeholder(tf.float32,[None,outputnum])
    keep_prob=tf.placeholder(tf.float32)
    Ws=[]
    bs=[]
    hiddens=[]
    drops=[x]
    for i in range(len(nodenums)-1):
        if(i==len(nodenums)-2):
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=10, stddev=0.1), name="W"+str(i)+cutime)
        else:
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=0, stddev=0.1), name="W"+str(i)+cutime)
        Ws.append(Wi)
        bi= tf.Variable(tf.ones(nodenums[i+1]), name="b"+str(i)+cutime)
        bs.append(bi)
        if i<len(nodenums)-2:
            hiddeni = tf.nn.relu(tf.add(tf.matmul(drops[i],Wi),bi))
            hiddens.append(hiddeni)
            dropi=tf.nn.dropout(hiddeni,keep_prob)
            drops.append(dropi)
        else:
            y=tf.add(tf.matmul(drops[i],Wi),bi)
    saver = tf.train.Saver()
    lossfun=tf.reduce_mean(tf.abs(tf.subtract(y_/y,1)))
    step = tf.Variable(0)
    train_step=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossfun)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(times):
        noiseX=npx+2*np.random.random(npx.shape)-1
        noiseY=npy+2*np.random.random(npy.shape)-1
        sess.run(train_step,feed_dict={x:noiseX,y_:noiseY,keep_prob:keep})
        loss1=sess.run(lossfun,feed_dict={x:npx,y_:npy,keep_prob:1})
        loss2=sess.run(lossfun,feed_dict={x:npx_test,y_:npy_test,keep_prob:1})
        losses1.append(loss1)
        losses2.append(loss2)
        losscol.append([loss1,loss2])
    losscolnp=np.array(losscol)
    save_path = saver.save(sess, path+modelname+".ckpt")
    lossplots.append(modelname+"nihe"+","+",".join(listmap(str,losses1[-15000:])))
    print(losses1[-1])
    lossplots.append(modelname+"yanzheng"+","+",".join(listmap(str,losses2[-15000:])))
    print(losses2[-1])
    preY=sess.run(y,feed_dict={x:npx_test,keep_prob:1})
    np.savetxt(path+"preY.csv",preY,fmt="%.8f",delimiter=',')
    np.savetxt(path+"losscol.csv",losscolnp,fmt="%.8f",delimiter=',')
    return losses2[-1]

def pred(hiddennum,outputnum,modelname,cutime,preX):
    inputnum=len(trainX[0])
    npx=np.array(trainX)
    npy=np.array(trainY)
    print(np.shape(npx))
    print(np.shape(npy))
    nodenums=[inputnum]+hiddennum+[outputnum]
    sess=tf.InteractiveSession()
    x=tf.placeholder(tf.float32, [None,inputnum])
    y_=tf.placeholder(tf.float32,[None,outputnum])
    keep_prob=tf.placeholder(tf.float32)
    Ws=[]
    bs=[]
    hiddens=[]
    drops=[x]
    for i in range(len(nodenums)-1):
        if(i==len(nodenums)-2):
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=10, stddev=0.1), name="W"+str(i)+cutime)
        else:
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=0, stddev=0.1), name="W"+str(i)+cutime)
        Ws.append(Wi)
        bi= tf.Variable(tf.ones(nodenums[i+1]), name="b"+str(i)+cutime)
        bs.append(bi)
        if i<len(nodenums)-2:
            hiddeni = tf.nn.relu(tf.add(tf.matmul(drops[i],Wi),bi))
            hiddens.append(hiddeni)
            dropi=tf.nn.dropout(hiddeni,keep_prob)
            drops.append(dropi)
        else:
            y=tf.add(tf.matmul(drops[i],Wi),bi)
    saver = tf.train.Saver()
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
trainX,trainY,validX,validY,preX=decompose(tensor,12,6,6,7)
print(trainX.shape,trainY.shape,validX.shape,validY.shape,preX.shape)

# onehide(trainX, trainY, 6, int(1e4), 0.94, taskname)
pred([8,8,8],6,taskname, cutime, preX)

# for nodenum in range(4,60):
# for i in range(10):
#     cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
#     print(cutime)
#     path = prepath + taskname + "\\" + cutime + "\\"
#     os.makedirs(path)
#     print(path)
#     # print(nodenum)
#     lossi=fcn(trainX, trainY, [8,8,8], int(5e4), 0.9, taskname, cutime, validX, validY)
#     with open(path +"lossplots"+datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")+".csv","w") as f:
#         f.write("\n".join(lossplots))
#     #     if(lossi<0.09):
#     #         break