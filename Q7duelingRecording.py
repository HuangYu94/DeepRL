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
# ConvW1 = np.reshape(np.loadtxt('Q7_P0_ConvW1.out'),[8,8,historyLen, out_dim1])
# ConvW2 = np.reshape(np.loadtxt('Q7_P0_ConvW2.out'),[4,4,out_dim1, out_dim2])
# ConvW3 = np.reshape(np.loadtxt('Q7_P0_ConvW3.out'),[3,3,out_dim2, out_dim3])
# ConvB1 = np.loadtxt('Q7_P0_ConvB1.out')
# ConvB2 = np.loadtxt('Q7_P0_ConvB2.out')
# ConvB3 = np.loadtxt('Q7_P0_ConvB3.out')
# VFC_w1 = np.loadtxt('Q7_P0_Value_FC_w1.out')
# VFC_w2 = np.loadtxt('Q7_P0_Value_FC_w2.out')
# VFC_b1 = np.loadtxt('Q7_P0_Value_FC_b1.out')
# VFC_b2 = np.loadtxt('Q7_P0_Value_FC_b2.out')
# AFC_w1 = np.loadtxt('Q7_P0_Advantage_FC_w1.out')
# AFC_w2 = np.loadtxt('Q7_P0_Advantage_FC_w2.out')
# AFC_b1 = np.loadtxt('Q7_P0_Advantage_FC_b1.out')
# AFC_b2 = np.loadtxt('Q7_P0_Advantage_FC_b2.out')
## phase1
ConvW1 = np.reshape(np.loadtxt('Q7_P1_ConvW1.out'),[8,8,historyLen, out_dim1])
ConvW2 = np.reshape(np.loadtxt('Q7_P1_ConvW2.out'),[4,4,out_dim1, out_dim2])
ConvW3 = np.reshape(np.loadtxt('Q7_P1_ConvW3.out'),[3,3,out_dim2, out_dim3])
ConvB1 = np.loadtxt('Q7_P1_ConvB1.out')
ConvB2 = np.loadtxt('Q7_P1_ConvB2.out')
ConvB3 = np.loadtxt('Q7_P1_ConvB3.out')
VFC_w1 = np.loadtxt('Q7_P1_Value_FC_w1.out')
VFC_w2 = np.loadtxt('Q7_P1_Value_FC_w2.out')
VFC_b1 = np.loadtxt('Q7_P1_Value_FC_b1.out')
VFC_b2 = np.loadtxt('Q7_P1_Value_FC_b2.out')
AFC_w1 = np.loadtxt('Q7_P1_Advantage_FC_w1.out')
AFC_w2 = np.loadtxt('Q7_P1_Advantage_FC_w2.out')
AFC_b1 = np.loadtxt('Q7_P1_Advantage_FC_b1.out')
AFC_b2 = np.loadtxt('Q7_P1_Advantage_FC_b2.out')

## phase2
# ConvW1 = np.reshape(np.loadtxt('Q7_P2_ConvW1.out'),[8,8,historyLen, out_dim1])
# ConvW2 = np.reshape(np.loadtxt('Q7_P2_ConvW2.out'),[4,4,out_dim1, out_dim2])
# ConvW3 = np.reshape(np.loadtxt('Q7_P2_ConvW3.out'),[3,3,out_dim2, out_dim3])
# ConvB1 = np.loadtxt('Q7_P2_ConvB1.out')
# ConvB2 = np.loadtxt('Q7_P2_ConvB2.out')
# ConvB3 = np.loadtxt('Q7_P2_ConvB3.out')
# VFC_w1 = np.loadtxt('Q7_P2_Value_FC_w1.out')
# VFC_w2 = np.loadtxt('Q7_P2_Value_FC_w2.out')
# VFC_b1 = np.loadtxt('Q7_P2_Value_FC_b1.out')
# VFC_b2 = np.loadtxt('Q7_P2_Value_FC_b2.out')
# AFC_w1 = np.loadtxt('Q7_P2_Advantage_FC_w1.out')
# AFC_w2 = np.loadtxt('Q7_P2_Advantage_FC_w2.out')
# AFC_b1 = np.loadtxt('Q7_P2_Advantage_FC_b1.out')
# AFC_b2 = np.loadtxt('Q7_P2_Advantage_FC_b2.out')
## phase3
# ConvW1 = np.reshape(np.loadtxt('Q7_P3_ConvW1.out'),[8,8,historyLen, out_dim1])
# ConvW2 = np.reshape(np.loadtxt('Q7_P3_ConvW2.out'),[4,4,out_dim1, out_dim2])
# ConvW3 = np.reshape(np.loadtxt('Q7_P3_ConvW3.out'),[3,3,out_dim2, out_dim3])
# ConvB1 = np.loadtxt('Q7_P3_ConvB1.out')
# ConvB2 = np.loadtxt('Q7_P3_ConvB2.out')
# ConvB3 = np.loadtxt('Q7_P3_ConvB3.out')
# VFC_w1 = np.loadtxt('Q7_P3_Value_FC_w1.out')
# VFC_w2 = np.loadtxt('Q7_P3_Value_FC_w2.out')
# VFC_b1 = np.loadtxt('Q7_P3_Value_FC_b1.out')
# VFC_b2 = np.loadtxt('Q7_P3_Value_FC_b2.out')
# AFC_w1 = np.loadtxt('Q7_P3_Advantage_FC_w1.out')
# AFC_w2 = np.loadtxt('Q7_P3_Advantage_FC_w2.out')
# AFC_b1 = np.loadtxt('Q7_P3_Advantage_FC_b1.out')
# AFC_b2 = np.loadtxt('Q7_P3_Advantage_FC_b2.out')


## construct deepQ network
sess = tf.Session()
inputState = tf.placeholder(dtype=tf.float32,shape=[1,3,84,84])
#=========BUILD ONLINE NETWORK========================
#========build 1st convolutional layer=========
online_ConvW1 = tf.Variable(tf.truncated_normal([8,8,historyLen, out_dim1],stddev = 0.1), name = 'online_ConvW1')
online_ConvB1 = tf.Variable(tf.truncated_normal([out_dim1],stddev=0.1), name = "online_ConvB1")
conv1 = tf.nn.conv2d(inputState, online_ConvW1, [1,1,4,4], padding='VALID', data_format='NCHW')
out1 = tf.nn.bias_add(conv1,online_ConvB1,data_format='NCHW')
conv_out1 = tf.nn.relu(out1)
#========build 2nd convolutional layer=========
online_ConvW2 = tf.Variable(tf.truncated_normal([4,4,out_dim1, out_dim2],stddev = 0.1), name = 'online_ConvW2')
online_ConvB2 = tf.Variable(tf.truncated_normal([out_dim2],stddev=0.1), name = 'online_ConvB2')
conv2 = tf.nn.conv2d(conv_out1,online_ConvW2,[1,1,2,2],padding='VALID', data_format='NCHW')
out2 = tf.nn.bias_add(conv2,online_ConvB2,data_format='NCHW')
conv_out2 = tf.nn.relu(out2)
#========build 3rd convolutional layer=========
online_ConvW3 = tf.Variable(tf.truncated_normal([3,3,out_dim2, out_dim3], stddev=0.1), name = 'online_ConvW3')
online_ConvB3 = tf.Variable(tf.truncated_normal([out_dim3],stddev=0.1), name = 'online_ConvB3')
conv3 = tf.nn.conv2d(conv_out2,online_ConvW3,[1,1,1,1],padding='VALID', data_format='NCHW')
out3 = tf.nn.bias_add(conv3,online_ConvB3,data_format='NCHW')
conv_out3 = tf.nn.relu(out3)
conv_out3_flat = tf.reshape(conv_out3,[-1, 7*7*64])
# #========build 1st fully connected layer of value function===========
# Value_FC_w1 = tf.Variable(tf.truncated_normal([7*7*64, 512],stddev=0.1),name='Value_FC_w1')
# Value_FC_b1 = tf.Variable(tf.truncated_normal([512],stddev=0.1),name='Value_FC_b1')
# Value_FC_out1 = tf.nn.relu(tf.matmul(conv_out3_flat, Value_FC_w1) + Value_FC_b1)
# #========build final fully connected layer of value function=========
# Value_FC_w2 = tf.Variable(tf.truncated_normal([512, 1],stddev=0.1), name = 'Value_FC_w2')
# Value_FC_b2 = tf.Variable(tf.truncated_normal([1],stddev=0.1), name = 'Value_FC_b2')
# Value_FC_out2 = tf.matmul(Value_FC_out1, Value_FC_w2) + Value_FC_b2
# I think we only need advantage to do decision making!!
#========build 1st fully connected layer of advantage function=======
Advantage_FC_w1 = tf.Variable(tf.truncated_normal([7*7*64,  512],stddev=0.1),name='Advantage_FC_w1')
Advantage_FC_b1 = tf.Variable(tf.truncated_normal([512],stddev=0.1),name='Advantage_FC_b1')
Advantage_FC_out1 = tf.matmul(conv_out3_flat, Advantage_FC_w1) + Advantage_FC_b1
#========build final fully connected layer of advantage function=====
Advantage_FC_w2 = tf.Variable(tf.truncated_normal([512, output_num_final],stddev=0.1),name='Advantage_FC_w2')
Advantage_FC_b2 = tf.Variable(tf.truncated_normal([output_num_final],stddev=0.1),name='Advantage_FC_b2')
Advantage_FC_out2 = tf.matmul(Advantage_FC_out1, Advantage_FC_w2)+Advantage_FC_b2

output_Q = Advantage_FC_out2


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
        feed = {inputState:[observationInput], online_ConvW1:ConvW1, online_ConvW2:ConvW2,
        online_ConvW3:ConvW3, online_ConvB1:ConvB1, online_ConvB2:ConvB2, online_ConvB3:ConvB3,
        Advantage_FC_w1:AFC_w1, Advantage_FC_w2:AFC_w2, Advantage_FC_b1:AFC_b1, Advantage_FC_b2:AFC_b2}
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