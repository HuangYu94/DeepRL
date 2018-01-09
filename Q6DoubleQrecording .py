import numpy as np
import tensorflow as tf
import cv2
import gym
from gym import wrappers

def imageMatrixParser(observation):
    temp_image = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    resizedImage = cv2.resize(temp_image, (84, 84), interpolation=cv2.INTER_AREA)
    return resizedImage




out_dim1 = 32 #number of filters for convolutional layer1
out_dim2 = 64 #number of filters for convolutional layer2
out_dim3 = 64 #number of filters for convolutional layer3
historyLen = 3
env = gym.make('SpaceInvadersNoFrameskip-v0')
env = wrappers.Monitor(env, '/10703-assignment2-YHuang-YZhou', force = True)
env.render()
output_num_final = env.action_space.n
## phase0
# Q1ConvW1 = np.reshape(np.loadtxt('Q6_P0_Q1_ConvW1.out'),[8,8,historyLen, out_dim1])
# Q1ConvW2 = np.reshape(np.loadtxt('Q6_P0_Q1_ConvW2.out'),[4,4,out_dim1, out_dim2])
# Q1ConvW3 = np.reshape(np.loadtxt('Q6_P0_Q1_ConvW3.out'),[3,3,out_dim2, out_dim3])
# Q1ConvB1 = np.loadtxt('Q6_P0_Q1_ConvB1.out')
# Q1ConvB2 = np.loadtxt('Q6_P0_Q1_ConvB2.out')
# Q1ConvB3 = np.loadtxt('Q6_P0_Q1_ConvB3.out')
# Q1FC_w1 = np.loadtxt('Q6_P0_Q1_FC_w1.out')
# Q1FC_w2 = np.loadtxt('Q6_P0_Q1_FC_w2.out')
# Q1FC_b1 = np.loadtxt('Q6_P0_Q1_FC_b1.out')
# Q1FC_b2 = np.loadtxt('Q6_P0_Q1_FC_b2.out')
# Q2ConvW1 = np.reshape(np.loadtxt('Q6_P0_Q2_ConvW1.out'),[8,8,historyLen, out_dim1])
# Q2ConvW2 = np.reshape(np.loadtxt('Q6_P0_Q2_ConvW2.out'),[4,4,out_dim1, out_dim2])
# Q2ConvW3 = np.reshape(np.loadtxt('Q6_P0_Q2_ConvW3.out'),[3,3,out_dim2, out_dim3])
# Q2ConvB1 = np.loadtxt('Q6_P0_Q2_ConvB1.out')
# Q2ConvB2 = np.loadtxt('Q6_P0_Q2_ConvB2.out')
# Q2ConvB3 = np.loadtxt('Q6_P0_Q2_ConvB3.out')
# Q2FC_w1 = np.loadtxt('Q6_P0_Q2_FC_w1.out')
# Q2FC_w2 = np.loadtxt('Q6_P0_Q2_FC_w2.out')
# Q2FC_b1 = np.loadtxt('Q6_P0_Q2_FC_b1.out')
# Q2FC_b2 = np.loadtxt('Q6_P0_Q2_FC_b2.out')
## phase1
Q1ConvW1 = np.reshape(np.loadtxt('Q6_P1_Q1_ConvW1.out'),[8,8,historyLen, out_dim1])
Q1ConvW2 = np.reshape(np.loadtxt('Q6_P1_Q1_ConvW2.out'),[4,4,out_dim1, out_dim2])
Q1ConvW3 = np.reshape(np.loadtxt('Q6_P1_Q1_ConvW3.out'),[3,3,out_dim2, out_dim3])
Q1ConvB1 = np.loadtxt('Q6_P1_Q1_ConvB1.out')
Q1ConvB2 = np.loadtxt('Q6_P1_Q1_ConvB2.out')
Q1ConvB3 = np.loadtxt('Q6_P1_Q1_ConvB3.out')
Q1FC_w1 = np.loadtxt('Q6_P1_Q1_FC_w1.out')
Q1FC_w2 = np.loadtxt('Q6_P1_Q1_FC_w2.out')
Q1FC_b1 = np.loadtxt('Q6_P1_Q1_FC_b1.out')
Q1FC_b2 = np.loadtxt('Q6_P1_Q1_FC_b2.out')
Q2ConvW1 = np.reshape(np.loadtxt('Q6_P1_Q2_ConvW1.out'),[8,8,historyLen, out_dim1])
Q2ConvW2 = np.reshape(np.loadtxt('Q6_P1_Q2_ConvW2.out'),[4,4,out_dim1, out_dim2])
Q2ConvW3 = np.reshape(np.loadtxt('Q6_P1_Q2_ConvW3.out'),[3,3,out_dim2, out_dim3])
Q2ConvB1 = np.loadtxt('Q6_P1_Q2_ConvB1.out')
Q2ConvB2 = np.loadtxt('Q6_P1_Q2_ConvB2.out')
Q2ConvB3 = np.loadtxt('Q6_P1_Q2_ConvB3.out')
Q2FC_w1 = np.loadtxt('Q6_P1_Q2_FC_w1.out')
Q2FC_w2 = np.loadtxt('Q6_P1_Q2_FC_w2.out')
Q2FC_b1 = np.loadtxt('Q6_P1_Q2_FC_b1.out')
Q2FC_b2 = np.loadtxt('Q6_P1_Q2_FC_b2.out')
## phase2
# Q1ConvW1 = np.reshape(np.loadtxt('Q6_P2_Q1_ConvW1.out'),[8,8,historyLen, out_dim1])
# Q1ConvW2 = np.reshape(np.loadtxt('Q6_P2_Q1_ConvW2.out'),[4,4,out_dim1, out_dim2])
# Q1ConvW3 = np.reshape(np.loadtxt('Q6_P2_Q1_ConvW3.out'),[3,3,out_dim2, out_dim3])
# Q1ConvB1 = np.loadtxt('Q6_P2_Q1_ConvB1.out')
# Q1ConvB2 = np.loadtxt('Q6_P2_Q1_ConvB2.out')
# Q1ConvB3 = np.loadtxt('Q6_P2_Q1_ConvB3.out')
# Q1FC_w1 = np.loadtxt('Q6_P2_Q1_FC_w1.out')
# Q1FC_w2 = np.loadtxt('Q6_P2_Q1_FC_w2.out')
# Q1FC_b1 = np.loadtxt('Q6_P2_Q1_FC_b1.out')
# Q1FC_b2 = np.loadtxt('Q6_P2_Q1_FC_b2.out')
# Q2ConvW1 = np.reshape(np.loadtxt('Q6_P2_Q2_ConvW1.out'),[8,8,historyLen, out_dim1])
# Q2ConvW2 = np.reshape(np.loadtxt('Q6_P2_Q2_ConvW2.out'),[4,4,out_dim1, out_dim2])
# Q2ConvW3 = np.reshape(np.loadtxt('Q6_P2_Q2_ConvW3.out'),[3,3,out_dim2, out_dim3])
# Q2ConvB1 = np.loadtxt('Q6_P2_Q2_ConvB1.out')
# Q2ConvB2 = np.loadtxt('Q6_P2_Q2_ConvB2.out')
# Q2ConvB3 = np.loadtxt('Q6_P2_Q2_ConvB3.out')
# Q2FC_w1 = np.loadtxt('Q6_P2_Q2_FC_w1.out')
# Q2FC_w2 = np.loadtxt('Q6_P2_Q2_FC_w2.out')
# Q2FC_b1 = np.loadtxt('Q6_P2_Q2_FC_b1.out')
# Q2FC_b2 = np.loadtxt('Q6_P2_Q2_FC_b2.out')
## phase3
# Q1ConvW1 = np.reshape(np.loadtxt('Q6_P3_Q1_ConvW1.out'),[8,8,historyLen, out_dim1])
# Q1ConvW2 = np.reshape(np.loadtxt('Q6_P3_Q1_ConvW2.out'),[4,4,out_dim1, out_dim2])
# Q1ConvW3 = np.reshape(np.loadtxt('Q6_P3_Q1_ConvW3.out'),[3,3,out_dim2, out_dim3])
# Q1ConvB1 = np.loadtxt('Q6_P3_Q1_ConvB1.out')
# Q1ConvB2 = np.loadtxt('Q6_P3_Q1_ConvB2.out')
# Q1ConvB3 = np.loadtxt('Q6_P3_Q1_ConvB3.out')
# Q1FC_w1 = np.loadtxt('Q6_P3_Q1_FC_w1.out')
# Q1FC_w2 = np.loadtxt('Q6_P3_Q1_FC_w2.out')
# Q1FC_b1 = np.loadtxt('Q6_P3_Q1_FC_b1.out')
# Q1FC_b2 = np.loadtxt('Q6_P3_Q1_FC_b2.out')
# Q2ConvW1 = np.reshape(np.loadtxt('Q6_P3_Q2_ConvW1.out'),[8,8,historyLen, out_dim1])
# Q2ConvW2 = np.reshape(np.loadtxt('Q6_P3_Q2_ConvW2.out'),[4,4,out_dim1, out_dim2])
# Q2ConvW3 = np.reshape(np.loadtxt('Q6_P3_Q2_ConvW3.out'),[3,3,out_dim2, out_dim3])
# Q2ConvB1 = np.loadtxt('Q6_P3_Q2_ConvB1.out')
# Q2ConvB2 = np.loadtxt('Q6_P3_Q2_ConvB2.out')
# Q2ConvB3 = np.loadtxt('Q6_P3_Q2_ConvB3.out')
# Q2FC_w1 = np.loadtxt('Q6_P3_Q2_FC_w1.out')
# Q2FC_w2 = np.loadtxt('Q6_P3_Q2_FC_w2.out')
# Q2FC_b1 = np.loadtxt('Q6_P3_Q2_FC_b1.out')
# Q2FC_b2 = np.loadtxt('Q6_P3_Q2_FC_b2.out')


## construct deepQ network
sess = tf.Session()
inputState = tf.placeholder(dtype=tf.float32,shape=[1,3,84,84])
#==================build Q1 network=================================
#========build 1st convolutional layer=========
Q1_ConvW1 = tf.placeholder(dtype = tf.float32,shape=[8,8,historyLen, out_dim1], name = 'Q1_ConvW1')
Q1_ConvB1 = tf.placeholder(dtype = tf.float32,shape = [out_dim1], name = "Q1_ConvB1")
Q1_conv1 = tf.nn.conv2d(inputState, Q1_ConvW1, [1,1,4,4], padding='VALID', data_format='NCHW')
Q1_out1 = tf.nn.bias_add(Q1_conv1,Q1_ConvB1,data_format='NCHW')
conv_Q1_out1 = tf.nn.relu(Q1_out1)
#========build 2nd convolutional layer=========
Q1_ConvW2 = tf.placeholder(dtype = tf.float32, shape=[4,4,out_dim1, out_dim2], name = 'Q1_ConvW2')
Q1_ConvB2= tf.placeholder(dtype = tf.float32,shape=[out_dim2], name = 'Q1_ConvB2')
Q1_conv2 = tf.nn.conv2d(conv_Q1_out1,Q1_ConvW2,[1,1,2,2],padding='VALID', data_format='NCHW')
Q1_out2 = tf.nn.bias_add(Q1_conv2,Q1_ConvB2,data_format='NCHW')
conv_Q1_out2 = tf.nn.relu(Q1_out2)
#========build 3rd convolutional layer=========
Q1_ConvW3 = tf.placeholder(dtype = tf.float32,shape=[3,3,out_dim2, out_dim3], name = 'Q1_ConvW3')
Q1_ConvB3 = tf.placeholder(dtype = tf.float32,shape=[out_dim3], name = 'Q1_ConvB3')
Q1_conv3 = tf.nn.conv2d(conv_Q1_out2,Q1_ConvW3,[1,1,1,1],padding='VALID', data_format='NCHW')
Q1_out3 = tf.nn.bias_add(Q1_conv3,Q1_ConvB3,data_format='NCHW')
Q1_conv_out3 = tf.nn.relu(Q1_out3)
Q1_conv_out3_flat = tf.reshape(Q1_conv_out3,[-1, 7*7*64])
#========build 1st fully connected layer===========
Q1_FC_w1 = tf.placeholder(dtype=tf.float32,shape=[7*7*64, 512],name='Q1_FC_w1')
Q1_FC_b1 = tf.placeholder(dtype=tf.float32,shape=[512],name='Q1_FC_b1')
FC_Q1_out1 = tf.nn.relu(tf.matmul(Q1_conv_out3_flat, Q1_FC_w1) + Q1_FC_b1)
#========build final fully connected layer=========
Q1_FC_w2 = tf.placeholder(dtype = tf.float32,shape=[512, output_num_final], name = 'Q1_FC_w2')
Q1_FC_b2 = tf.placeholder(dtype = tf.float32, shape=[output_num_final], name = 'Q1_FC_b2')
output_Q1 = tf.matmul(FC_Q1_out1, Q1_FC_w2) + Q1_FC_b2

#==================build Q2 network================================
#========build 1st convolutional layer=========
Q2_ConvW1 = tf.placeholder(dtype = tf.float32,shape=[8,8,historyLen, out_dim1], name = 'Q2_ConvW1')
Q2_ConvB1 = tf.placeholder(dtype = tf.float32,shape = [out_dim1], name = "Q2_ConvB1")
Q2_conv1 = tf.nn.conv2d(inputState, Q2_ConvW1, [1,1,4,4], padding='VALID', data_format='NCHW')
Q2_out1 = tf.nn.bias_add(Q2_conv1,Q2_ConvB1,data_format='NCHW')
conv_Q2_out1 = tf.nn.relu(Q2_out1)
#========build 2nd convolutional layer=========
Q2_ConvW2 = tf.placeholder(dtype = tf.float32, shape=[4,4,out_dim1, out_dim2], name = 'Q2_ConvW2')
Q2_ConvB2= tf.placeholder(dtype = tf.float32,shape=[out_dim2], name = 'Q2_ConvB2')
Q2_conv2 = tf.nn.conv2d(conv_Q2_out1,Q2_ConvW2,[1,1,2,2],padding='VALID', data_format='NCHW')
Q2_out2 = tf.nn.bias_add(Q2_conv2,Q2_ConvB2,data_format='NCHW')
conv_Q2_out2 = tf.nn.relu(Q2_out2)
#========build 3rd convolutional layer=========
Q2_ConvW3 = tf.placeholder(dtype = tf.float32,shape=[3,3,out_dim2, out_dim3], name = 'Q2_ConvW3')
Q2_ConvB3 = tf.placeholder(dtype = tf.float32,shape=[out_dim3], name = 'Q2_ConvB3')
Q2_conv3 = tf.nn.conv2d(conv_Q2_out2,Q2_ConvW3,[1,1,1,1],padding='VALID', data_format='NCHW')
Q2_out3 = tf.nn.bias_add(Q2_conv3,Q2_ConvB3,data_format='NCHW')
Q2_conv_out3 = tf.nn.relu(Q2_out3)
Q2_conv_out3_flat = tf.reshape(Q2_conv_out3,[-1, 7*7*64])
#========build 1st fully connected layer===========
Q2_FC_w1 = tf.placeholder(dtype=tf.float32,shape=[7*7*64, 512],name='Q2_FC_w1')
Q2_FC_b1 = tf.placeholder(dtype=tf.float32,shape=[512],name='Q2_FC_b1')
FC_Q2_out1 = tf.nn.relu(tf.matmul(Q2_conv_out3_flat, Q2_FC_w1) + Q2_FC_b1)
#========build final fully connected layer=========
Q2_FC_w2 = tf.placeholder(dtype = tf.float32,shape=[512, output_num_final], name = 'Q2_FC_w2')
Q2_FC_b2 = tf.placeholder(dtype = tf.float32, shape=[output_num_final], name = 'Q2_FC_b2')
output_Q2 = tf.matmul(FC_Q2_out1, Q2_FC_w2) + Q2_FC_b2

output_Q = tf.add(output_Q1,output_Q2)

observation = env.reset() #observation initialization
action = env.action_space.sample() #action initialization
count = 1 #3 observation determine 1 action
observationInput = []
while True:
    imageMatrix = imageMatrixParser(observation)    
    observationInput.append(imageMatrix)
    if (count == 3):
        obsMatrix = np.asarray(observationInput, dtype = np.float32).reshape([1,-1])
        # actionMatrix = np.add(np.matmul(obsMatrix, weightData), biasData) #linearQ2Q3 action selector
        feed = {inputState:[observationInput], Q1_ConvW1:Q1ConvW1, Q1_ConvW2:Q1ConvW2,
         Q1_ConvW3:Q1ConvW3, Q1_ConvB1:Q1ConvB1, Q1_ConvB2:Q1ConvB2, Q1_ConvB3:Q1ConvB3,
          Q2_ConvW1:Q2ConvW1, Q2_ConvW2:Q2ConvW2, Q2_ConvW3:Q2ConvW3, Q2_ConvB1:Q2ConvB1,
           Q2_ConvB2:Q2ConvB2, Q2_ConvB3:Q2ConvB3, Q1_FC_w1:Q1FC_w1, Q1_FC_w2:Q1FC_w2,
            Q1_FC_b1:Q1FC_b1, Q1_FC_b2:Q1FC_b2, Q2_FC_w1:Q2FC_w1, Q2_FC_w2:Q2FC_w2, 
            Q2_FC_b1:Q2FC_b1, Q2_FC_b2:Q2FC_b2}
        actionMatrix = sess.run(output_Q, feed_dict=feed)#deepQ action selector
        action = np.argmax(actionMatrix)
        count = 1 #reset count
        observationInput = [] #reset observation
    else:
        count += 1
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()
        break

env.close()
gym.upload('/10703-assignment2-YHuang-YZhou', api_key='sk_tkcjGIDjS1K6ch6y0grpQ')