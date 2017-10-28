# Model for Modulation Recognition
# written by Saurabh Farkya
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np
from readMatfile import readMatfile
import os
from datetime import datetime
from radioNet import radioNet
from radioNetv2 import radioNetv2
from radioNetv3 import radioNetv3
import time

## Network parameter
# Learning parameters
batch_size = 1024
learning_rate = .0001
epoch_num = 50

# Network Parameters
dropout_rate = 0.5
num_classes = 5

steps_per_epoch = 1133
str_ad = "/media/saurabh/New Volume"
#str_ad = "/home/cwnlab"
# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = str_ad + "/RadioML/code/data/m10"#"/home/saurabh/deep_learning/project_execution/models_final/data"
checkpoint_path = str_ad + "/RadioML/code/checkpoint/m10"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
if not os.path.isdir(checkpoint_path): os.mkdir(filewriter_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None,2,128,1])
ypred = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# importing the model!!
y = radioNetv3(x,keep_prob)

# model done
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)

## cost function - cross entropy
# training
cross_entropy_training = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=ypred, logits=y))

# training
# cross_entropy_training_per_epoch = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=ypred, logits=y))

# validation
cross_entropy_validation = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=ypred, logits=y))


# adding to summary
sum_loss_valid = tf.summary.scalar("validation cost",cross_entropy_validation)
sum_loss_train = tf.summary.scalar("training cost",cross_entropy_training)
#tf.summary.scalar("training cost per epoch",cross_entropy_training_per_epoch)

## defining optimizer
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_training)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ypred, 1))

# training accuracy
accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# validation accuracy
accuracy_validation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training accuracy per epoch
#accuracy_train_per_epoch = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# variable for saving
accuracy_overall = np.zeros([1133, 1])
loss_overall = np.zeros([1133, 1])

# adding to summary
sum_acc_train = tf.summary.scalar("training_accuracy", accuracy_train)
sum_acc_valid = tf.summary.scalar("validation_accuracy",accuracy_validation)
#tf.summary.scalar("training accuracy per epoch",accuracy_train_per_epoch)
#tf.summary.scalar("Mean accuracy", )

# addding to all the summary
sum_train = tf.summary.merge([sum_acc_train,sum_loss_train])
sum_valid = tf.summary.merge([sum_acc_valid,sum_loss_valid])
## reading the whole data from the dataset.
# getting training and validation data

# parameters for saving graph
training_writer = tf.summary.FileWriter(filewriter_path + '/train',graph = tf.get_default_graph()) # add '/train
validation_writer = tf.summary.FileWriter(filewriter_path + '/validation' ) # add '/validation'

# Fetching the data!
x_train, y_train = readMatfile('train')#np.ones([1160000, 2, 128]),np.ones((1, 1160000))#readMatfile('train')
x_validation, y_validation = readMatfile('validation')#np.ones([20000, 2, 128, 1]),np.ones((1, 20000))#readMatfile('validation')
x_validation = np.reshape(x_validation,[20000,2,128,1])

# converting into one hot representation
sess = tf.Session()
y_train = sess.run(tf.one_hot(y_train,10))
y_validation = sess.run(tf.one_hot(y_validation,10))
y_train = np.reshape(y_train,[1160000,10])
y_validation = np.reshape(y_validation,[20000,10])

# Initialize an saver for store model checkpoints
saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)


with tf.Session() as sess:
    ## initialising the variables:
    sess.run(tf.global_variables_initializer())
  #  saver.restore(sess,"/media/saurabh/New Volume/RadioML/code/checkpoint/model_epoch199.ckpt")
    training_writer.add_graph(sess.graph)

    for epoch in range(0,epoch_num):
        #print('Epoch Number: ', epoch)
        start_time = time.time()
        training_indexes = np.arange(1160000)
        validation_indexes = np.arange(20000)
        np.random.shuffle(training_indexes)
        np.random.shuffle(validation_indexes)
        training_indexes = np.reshape(training_indexes,[1160000,1])
        validation_indexes = np.reshape(validation_indexes,[20000,1])
        step = 0
    #    print training_indexes.shape

        while step < steps_per_epoch:
            #print training_indexes[step*1024:(step+1)*1024-1,0]
            #print x_train.shape
            print('step %d' %(step))
            if step < 1132:
                batchx = x_train[training_indexes[step*1024:(step+1)*1024,0],:,:]
                batchy = y_train[training_indexes[step*1024:(step+1)*1024,0],:]
                batchx = np.reshape(batchx,[batch_size,2,128,1])
                #print batchx.shape, batchy.shape
            else:
                batchx = x_train[training_indexes[-823:,0],:,:]
                batchy = y_train[training_indexes[-823:, 0], :]
                batchx = np.reshape(batchx, [823, 2, 128, 1])

            summary_training, trainingLoss, training_accuracy,_ = sess.run(
                [sum_train, cross_entropy_training, accuracy_train, train_step],
                feed_dict={x: batchx, ypred: batchy, keep_prob: dropout_rate})

            #s_training = sess.run(summary, feed_dict={x: batchx, ypred: batchy, keep_prob: dropout_rate})
            # write log
            training_writer.add_summary(summary_training,epoch*steps_per_epoch+step)
            accuracy_overall[step, 0] = training_accuracy
            loss_overall[step, 0] = trainingLoss

            #training_accuracy = accuracy_train.eval(feed_dict= {x:batchx,ypred:batchy,keep_prob:1})

            timex = start_time - time.time()
            #print('training_accuracy in step %d is %g and loss is %g' %(step,training_accuracy,trainingLoss))
            step += 1

        # writing log of training accuracy per epoch
        # mean training accuracy
        meanAccuracy = np.mean(accuracy_overall)
        meanLoss = np.mean(loss_overall)
       # print('Mean training_accuracy after epoch %d is %g and loss is %g' % (epoch, meanAccuracy,meanLoss))

       # training_writer.add_summary(meanAccuracy,epoch)
        # writing accuracy to tensorboard
       # writer.add_summary()

        # For validation
        validation_accuracy = np.zeros([20,1])
        validationLoss = np.zeros([20,1])
        for i in range(20):

            if i < 19:
                #print(x_validation.shape, y_validation.shape)
                validationbatchx = x_validation[validation_indexes[i*1024:(i+1)*1024,0],:,:]
                validationbatchx = np.reshape(validationbatchx,[1024,2,128,1])
                validationbatchy = y_validation[validation_indexes[i*1024:(i+1)*1024,0],:]
                validationbatchy = np.reshape(validationbatchy,[1024,10])
                # mean validation set accuracy!
                 #validation_accuracy = accuracy_validation.eval(feed_dict={x: validationbatchx,ypred : validationbatchy,keep_prob : 1 })
            else:
                batchx = x_train[training_indexes[-544:, 0], :, :]
                batchy = y_train[training_indexes[-544:, 0], :]
                batchx = np.reshape(batchx, [544, 2, 128, 1])

            summary_validation, tempvalidation_accuracy, tempvalidationLoss = sess.run([sum_valid, accuracy_validation,cross_entropy_validation],
                                                               feed_dict={x: validationbatchx, ypred: validationbatchy,
                                                                          keep_prob: 1})

            validation_accuracy[i,0] = tempvalidation_accuracy
            validationLoss[i,0] = tempvalidationLoss
            validation_writer.add_summary(summary_validation,epoch*1133 +i)

        validationLoss = np.mean(validationLoss)
        validation_accuracy = np.mean(validation_accuracy)

        print('Epoch %d: Training accuracy = %g loss = %g || Validation accuracy %g loss = %g in time %f s' % (epoch, meanAccuracy, meanLoss, validation_accuracy,validationLoss,timex))
        ## saving the checkpoint!
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)
       # print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


## some points to be added to improvise it.
# show loss function for both training and validation set.
# also add this to the tensorboard link - https://www.tensorflow.org/get_started/summaries_and_tensorboard
# also add both training and validation accuracy!
# write code for mean validation accuracy.
# train the model on the old small data set and show the accuracy!
#