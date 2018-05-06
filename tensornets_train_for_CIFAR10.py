import tensorflow as tf
import numpy as np
import math
#import matplotlib.pyplot as plt
#%matplotlib inline
import time
import os
import tensornets as nets
import cv2
import sys
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000, cifar10_dir = None):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    #cifar10_dir = 'cs231n/datasets'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_img_from_tensor(x, target_size=None, crop_size=None, interp=cv2.INTER_CUBIC):

    minSize = min(x.shape[1:3])
    imgs = None
    if target_size:
        if isinstance(target_size, int):
            hw_tuple = (x.shape[1] * target_size // minSize, x.shape[2] * target_size // minSize)
        else:
            hw_tuple = (target_size[1], target_size[0])
        imgs = np.zeros((x.shape[0],hw_tuple[0],hw_tuple[1], 3), dtype=np.uint8)
        if x.shape[1:3] != hw_tuple:
            for i in range(x.shape[0]):
                imgs[i,:, :, :] = cv2.resize(x[i, :, :, :], hw_tuple, interpolation=interp)
    if crop_size is not None:
        imgs = nets.utils.crop(imgs, crop_size)
        
    return imgs

def run_model(session, Xd, yd, Xv, yv, num_class = 10, epochs=3, batch_size=100,print_every=10, learning_rate = 1e-5, dropout = 0.5):
    print("Batch dataset initialized.\n# of training data: {}\n# of test data: {}\n# of class: {}"
          .format(Xd.shape[0], Xv.shape[0], 10))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
        
    with tf.Session() as sess:

        inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        outputs = tf.placeholder(tf.int32, [None])
        
        cnn_net = nets.MobileNet100(inputs, is_training = True, classes = num_class)
        
        cnn_loss = tf.losses.softmax_cross_entropy(tf.one_hot(outputs,num_class, dtype=tf.int32), cnn_net)
        cnn_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cnn_loss)
        
        cnn_predictions = tf.argmax(cnn_net, axis = 1)
        cnn_correct_prediction = tf.equal(tf.cast(cnn_predictions, dtype=tf.int32), outputs)
        cnn_accuracy = tf.reduce_mean(tf.cast(cnn_correct_prediction, tf.float32))

        train_summary = tf.summary.merge([tf.summary.scalar("train_loss", cnn_loss),
                          tf.summary.scalar("train_accuracy", cnn_accuracy)])
                    
        test_summary = tf.summary.merge([tf.summary.scalar("val_loss", cnn_loss),
                          tf.summary.scalar("val_accuracy", cnn_net)])    
        
        sess.run(tf.global_variables_initializer())
        nets.pretrained(cnn_net)        
        
        # tensorboard setting

        fileName = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        fileName = os.path.normcase("./result/" + fileName)
        summary_writer = tf.summary.FileWriter(fileName, sess.graph)
        
        global_step = 0
                  
        for current_epoch in range(epochs):
            # training step
            ###for x_batch, y_batch in batch_set.batches():
            print("#############################Epoch Start##############################")
            
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                start = time.time()
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = np.int32(train_indicies[start_idx:start_idx+batch_size])
                                      
                batch_Xd = load_img_from_tensor(Xd[idx,:, :, :], target_size=256, crop_size=224)
                batch_Xd = cnn_net.preprocess(batch_Xd) 
                batch_yd = yd[idx]
                feed = {inputs : batch_Xd, outputs : batch_yd}                
                
                global_step = global_step + 1
                


                _, loss, scores,accuracy, summary = sess.run([cnn_train, cnn_loss, 
                                                              cnn_net, cnn_accuracy, train_summary], feed_dict=feed)
                
                summary_writer.add_summary(summary, global_step)

                
                if global_step % print_every == 0:
                    print("{}/{} ({} epochs) step, loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                          .format(global_step, int(round(Xd.shape[0]/batch_size)) * epochs, current_epoch,
                                  loss, accuracy, time.time() - start))

            # test step
            start, avg_loss, avg_accuracy = time.time(), 0, 0
            
            Xv = load_img_from_tensor(Xv, target_size=256, crop_size=224)
            Xv = cnn_net.preprocess(Xv) 
            feed = {inputs : Xv, outputs : yv}
            loss, accuracy, summary = sess.run([cnn_loss, cnn_accuracy, test_summary], feed_dict=feed)

            summary_writer.add_summary(summary, current_epoch)
            print("{} epochs test result. loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                  .format(current_epoch, loss , accuracy , time.time() - start))
            
            print("\n")
    return 



    
def main(_):
    if sys.platform == "linux" :
        cifar10_dir = "/home/z_tomcato/cs231n/assignment2/assignment2/cs231n/datasets/cifar-10-batches-py"
    else:
        cifar10_dir = 'cs231n/datasets'
    print("========================" + time.strftime("%Y%m%d_%H:%M:%S", time.localtime()) + "=========================")
    # Invoke the above function to get our data.
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(cifar10_dir = cifar10_dir)
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        #with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        #sess.run(tf.global_variables_initializer())
        #print('Training')
        run_model(sess, X_train[:5000], y_train[:5000],X_val,y_val, epochs=1, batch_size=500,print_every=10, learning_rate = 0.0001)
    print("==================================================================")
    print("\n")

if __name__ == '__main__':
    tf.app.run()