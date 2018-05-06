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
from load_wiki_cropface import get_wiki_crop_data

def run_model(session, Xd, yd, Xv, yv, num_class = 10, epochs=3, batch_size=100,print_every=10, 
              learning_rate = 1e-5, dropout = 0.5, is_save_summary = True, name = "for_wiki"):
    print("Batch dataset initialized.\n# of training data: {}\n# of val data: {}\n# of class: {}"
          .format(Xd.shape[0], Xv.shape[0], num_class))
    
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
                          tf.summary.scalar("val_accuracy", cnn_accuracy)])    
        
        sess.run(tf.global_variables_initializer())
        nets.pretrained(cnn_net)        
        
        # tensorboard setting

        if is_save_summary:
            fileName = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            fileName = os.path.normcase("./result/" + fileName)
            summary_writer = tf.summary.FileWriter(fileName, sess.graph)
        
        global_step = 0
                  
        saver = tf.train.Saver()

        for current_epoch in range(epochs):
            # training step
            ###for x_batch, y_batch in batch_set.batches():
            print("#############################Epoch Start##############################")
            
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                start = time.time()
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = np.int32(train_indicies[start_idx:start_idx+batch_size])
                                      
                #batch_Xd = load_img_from_tensor(Xd[idx,:, :, :], target_size=256, crop_size=224)
                batch_Xd = Xd[idx,:, :, :]
                batch_Xd = cnn_net.preprocess(batch_Xd)
                batch_yd = yd[idx]
                feed = {inputs : batch_Xd, outputs : batch_yd}                
                
                global_step = global_step + 1
                
                
                _, loss, scores,accuracy, summary = sess.run([cnn_train, cnn_loss,
                                           cnn_net, cnn_accuracy, train_summary], feed_dict=feed)

                if is_save_summary:
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

            if is_save_summary:
                summary_writer.add_summary(summary, current_epoch)
            print("{} epochs test result. loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                  .format(current_epoch, loss , accuracy , time.time() - start))
            
            print("\n")
        saver.save(sess, 'train_models/tensornets_wiki_cropface_model_' + name)
    return 


tf.app.flags.DEFINE_integer('batch_size', 200, 'batch size')
tf.app.flags.DEFINE_integer('num_train_data', None, 'number of train data')
tf.app.flags.DEFINE_integer('num_val_data', None, 'number of val data')
tf.app.flags.DEFINE_integer('num_epochs', 4, 'number of epochs')
tf.app.flags.DEFINE_float('learning_rate', 0.4, 'init learning rate')

tf.app.flags.DEFINE_integer('print_every', 5, 'how often to print training status')
tf.app.flags.DEFINE_boolean('is_save_summary', True, 'is save summary data')

FLAGS = tf.app.flags.FLAGS

def main(_):
    
    print("========================" + time.strftime("%Y%m%d_%H:%M:%S", time.localtime()) + "=========================")
    # Invoke the above function to get our data.
    age_dict, gender_dict = get_wiki_crop_data(num_training=49000, num_validation=1000, num_test=0)
    print('Train data shape: ', age_dict["X_train"].shape)
    print('Train labels shape: ', age_dict["y_train"].shape)
    print('Validation data shape: ', age_dict["X_val"].shape)
    print('Validation labels shape: ', age_dict["y_val"].shape)
    print('Test data shape: ', age_dict["X_test"].shape)
    print('Test labels shape: ', age_dict["y_test"].shape)
    
    """  
    print("train_wiki_age")
    tf.reset_default_graph()
    
   
    with tf.Session() as sess:
        #with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        #sess.run(tf.global_variables_initializer())
        #print('Training')
        run_model(sess, age_dict["X_train"][:FLAGS.num_train_data], age_dict["y_train"][:FLAGS.num_train_data],
                  age_dict["X_val"][:FLAGS.num_val_data], age_dict["y_val"][:FLAGS.num_val_data], 
                  num_class = 100, epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
                  print_every=FLAGS.print_every, learning_rate = FLAGS.learning_rate, 
                  is_save_summary = FLAGS.is_save_summary, name = "age_train")
        pass
    
    """
    
    print("train_wiki_gender")
    tf.reset_default_graph()
  
    with tf.Session() as sess:
        #with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        #sess.run(tf.global_variables_initializer())
        #print('Training')
        run_model(sess, gender_dict["X_train"][:FLAGS.num_train_data], gender_dict["y_train"][:FLAGS.num_train_data],
                  gender_dict["X_val"][:FLAGS.num_val_data], gender_dict["y_val"][:FLAGS.num_val_data], 
                  num_class = 2, epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
                  print_every=FLAGS.print_every, learning_rate = FLAGS.learning_rate, 
                  is_save_summary = FLAGS.is_save_summary, name = "gender_train")
        pass
    print("==================================================================")
    print("\n")

if __name__ == '__main__':
    tf.app.run()