import gym
import tensorflow as tf
import numpy as np
import random
import policy
import objectives
import copy
import cv2
# This file implements the problem6 double Q network with experience replay and
# and target fixing to the extent possible as describled in reference 3
# I will try to make the parameter setting similiar to reference 3to the 
# extent possible!
#===================utility classes are defined as follows===================
class Sample:
    #This class store one sample of the tuple (s,r,a,s`,terminal)
    # Follows are static variable for the class,, they are:
    # history length, height, width.
    historyLen = 3
    height = 84
    width = 84
    def __init__(self, state_in, action_in, reward_in, nextstate_in, is_terminal):
        #input states, action, reward are all lists of 3 frames since I use the
        #no frame skipping version of SpaceInvaders
        self.state = state_in
        self.nextstate = nextstate_in
        self.action = action_in
        self.terminal = is_terminal
        reward = sum(reward_in)
        self.reward = reward

    def getProcessedState(self):
        # Make the sample input to CNN and the format here is [history_length,
        # height, width] which is [3,84,84] specifically.
        #We convert the uint8 data type to float32 type to let the neural
        #network better process the image
        #return: processed image as float32
        temp = []
        for oneframe in self.state:
            tempFrame = ImagePreprocess(oneframe)
            temp.append(copy.deepcopy(tempFrame))
        processedState = np.asarray(temp, dtype=np.float32)
        processedState = processedState.reshape([self.historyLen, self.height, self.width])
        return [processedState]
    
    def getProcessedNextState(self):
        #similiar to the previous function
        #this just return the processed next state stored in the sample
        temp = []
        for oneframe in self.nextstate:
            tempFrame = ImagePreprocess(oneframe)
            temp.append(copy.deepcopy(tempFrame))
        processedNextState = np.asarray(temp, dtype=np.float32)
        processedNextState = processedNextState.reshape([self.historyLen, self.height, self.width])
        return [processedNextState]

        
    def getSample(self):
        return (self.state, self.action, self.reward, self.nextstate)
    
    def getState(self):
        return self.state
    
    def getAction(self):
        return np.asarray(self.action,dtype=np.int32)
        
    def getNextState(self):
        return self.nextstate
        
    def getReward(self):
        return [np.asarray(self.reward,dtype=np.float32)]
        
    def getTerminal(self):
        return [float(self.terminal)]
        

class Experience:
    #this class is the sample pool mentioned in the paper which stores multiple
    #samples and we extract randomly from this class to create a memory replay 
    #training batch used in the algorithm
    #@PARAM:
    #SamplePool: list to store all samples
    def __init__(self):
        #create empty list to store the samples
        self.SamplePool = []
        self.replayBufferSize = 100000#change this to 1000000 when submit
        
    def __getTrainingBatch(self, env, sess, batch_size, output_Q, p):
        #Make the first batch if current sample pool have samples less
        #than required batchsize.
        #You should not call the function outside the class, so I make it 
        #private!
        stateBatch = []
        nextStateBatch = []
        actionBatch = []
        rewardBatch = []
        terminalBatch = []
        samples = []
        for i in range(batch_size):
            if i == 0:
                action = env.action_space.sample()
                frame1, reward1, is_terminal1, info = env.step(action)
                frame2, reward2, is_terminal2, info = env.step(action)
                frame3, reward3, is_terminal3, info = env.step(action)
                # frame4, reward4, is_terminal4, info = env.step(action)
                # state = [np.copy(frame1), np.copy(frame2), np.copy(frame3),
                # np.copy(frame4)]
                state = [np.copy(frame1), np.copy(frame2), np.copy(frame3)]
                aciton = env.action_space.sample()
                frame5, reward5, is_terminal5, info = env.step(action)
                frame6, reward6, is_terminal6, info = env.step(action)
                frame7, reward7, is_terminal7, info = env.step(action)
                # frame8, reward8, is_terminal8, info = env.step(action)
                # nextState = [np.copy(frame5), np.copy(frame6),
                # np.copy(frame7), np.copy(frame8)]
                nextState = [np.copy(frame5), np.copy(frame6), np.copy(frame7)]
                # terminal = max([is_terminal5, is_terminal6, is_terminal7,
                # is_terminal8])
                terminal = max([is_terminal5, is_terminal6, is_terminal7])
                # reward = [reward5, reward6, reward7, reward8]
                reward = [reward5, reward6, reward7]
                tempSample = Sample(state, action, reward,nextState, terminal)
                stateBatch.append(copy.deepcopy(
                tempSample.getProcessedState()[0]))
                nextStateBatch.append(copy.deepcopy(
                tempSample.getProcessedNextState()[0]))
                actionBatch.append([i,tempSample.getAction()])
                rewardBatch.append(tempSample.getReward())
                samples.append(copy.deepcopy(tempSample))
                terminalBatch.append(tempSample.getTerminal())
                last_state = copy.deepcopy(tempSample.getProcessedNextState())
                if terminal == 1:
                    env.reset()
                self.SamplePool.append(copy.deepcopy(tempSample))
            else:
                q_value = output_Q.eval(feed_dict = {x1:last_state,
                x2:last_state},
                 session = sess)
                action = p.select_action(q_value)
                frame5, reward5, is_terminal5, info = env.step(action)
                frame6, reward6, is_terminal6, info = env.step(action)
                frame7, reward7, is_terminal7, info = env.step(action)
                # frame8, reward8, is_terminal8, info = env.step(action)
                state = samples[i-1].getNextState()
                # nextState = [np.copy(frame5), np.copy(frame6),
                # np.copy(frame7), np.copy(frame8)]
                nextState = [np.copy(frame5), np.copy(frame6), np.copy(frame7)]
                # terminal = max([is_terminal5, is_terminal6, is_terminal7,
                # is_terminal8])
                terminal = max([is_terminal5, is_terminal6, is_terminal7])
                # reward = [reward5, reward6, reward7, reward8]
                reward = [reward5, reward6, reward7]
    
                tempSample = Sample(state, action, reward, nextState, terminal)
                stateBatch.append(copy.deepcopy(
                tempSample.getProcessedState()[0]))
                nextStateBatch.append(copy.deepcopy(
                tempSample.getProcessedNextState()[0]))
                actionBatch.append([i,tempSample.getAction()])
                rewardBatch.append(tempSample.getReward())
                terminalBatch.append(tempSample.getTerminal())
                samples.append(copy.deepcopy(tempSample))
                last_state = copy.deepcopy(tempSample.getProcessedNextState())
                if terminal == 1:
                    env.reset()
                self.SamplePool.append(copy.deepcopy(tempSample))
                
                
        return (stateBatch, actionBatch, rewardBatch,\
        nextStateBatch,terminalBatch)


    def addSample(self, sample):
        #add one sample to the sample list
        if len(self.SamplePool) < self.replayBufferSize:
            self.SamplePool.append(copy.deepcopy(sample))
        else:
            self.SamplePool.remove(self.SamplePool[0])
            self.SamplePool.append(copy.deepcopy(sample))
        
        
    def getRandomBatch(self, env, sess, batch_size, output_Q, p):
        #the way of sampling random batch is two-fold:
        #1.if current length of sample pool is smaller than the given batchsize
        #I randomly create the minimal number of samples to return and and that 
        #to the sample pool
        #2. if current length of sample pool is bigger than the given batchsize
        #then it`s the time for me to do the sampling describled in the papers
        #RETURN:stateBatch, actionBatch, rewardBatch, nextStateBatch,
        #terminalBatch
        #Initialize the sample list to be returned:
        stateBatch = []
        actionBatch = []
        rewardBatch = []
        nextStateBatch = []
        terminalBatch = []
        if len(self.SamplePool) < batch_size:
            stateBatch, actionBatch, rewardBatch, nextStateBatch,terminalBatch= self.__getTrainingBatch(env, sess, batch_size, output_Q, p)
            return (stateBatch,actionBatch,rewardBatch,nextStateBatch, \
            terminalBatch)
        else:
            indexList = random.sample(range(0, len(self.SamplePool)), batch_size)
            for i,randIndex in enumerate(indexList):
                tempSample = copy.deepcopy(self.SamplePool[randIndex])
                stateBatch.append(copy.deepcopy(
                tempSample.getProcessedState()[0]))
                actionBatch.append([i,copy.deepcopy(tempSample.getAction())])
                rewardBatch.append(copy.deepcopy(tempSample.getReward()))
                nextStateBatch.append(copy.deepcopy(
                tempSample.getProcessedNextState()[0]))
                terminalBatch.append(copy.deepcopy(tempSample.getTerminal()))

            return (stateBatch,actionBatch,rewardBatch, \
            nextStateBatch,terminalBatch)
                
    def getLastState(self):
        #Let the instance to send the last state which is s_t so that we
        #can use ExecuteOneStep to get the next state and make the sample
        #RETURN
        #retPrecessed: processed state to be the input of ExecuteOneStep()
        #retRaw: unprocessed rawstate to set the returned sample after execute
        #one step
        lastIndex = len(self.SamplePool)-1
        lastSample = self.SamplePool[lastIndex]
        retProcessed = copy.deepcopy(lastSample.getProcessedNextState())
        retRaw = copy.deepcopy(lastSample.getNextState())
        return retProcessed, retRaw
    
    def reset(self):
        #reset experience instance wipe out all samples!
        self.SamplePool = []


#==================utility functions are defined here===================
def Q6Saver(Q1_ConvW1, Q1_ConvB1, Q1_ConvW2, Q1_ConvB2, Q1_ConvW3, Q1_ConvB3, Q1_FC_w1, Q1_FC_b1, Q1_FC_w2, Q1_FC_b2, Q2_ConvW1, Q2_ConvB1, Q2_ConvW2, Q2_ConvB2, Q2_ConvW3, Q2_ConvB3, Q2_FC_w1, Q2_FC_b1, Q2_FC_w2, Q2_FC_b2, trainPhase, sess):
    # same as before store all the weights and flatten those weights in
    # convolutional layers
    questionNum = 'Q6'
    phase = '_' + trainPhase + '_'
    np.savetxt(questionNum+phase+'Q1_ConvW1.out', np.reshape(Q1_ConvW1.eval(session = sess),[-1]))
    np.savetxt(questionNum+phase+'Q1_ConvB1.out', Q1_ConvB1.eval(session=sess))
    np.savetxt(questionNum+phase+'Q1_ConvW2.out', np.reshape(Q1_ConvW2.eval(session = sess),[-1]))
    np.savetxt(questionNum+phase+'Q1_ConvB2.out', Q1_ConvB2.eval(session=sess))
    np.savetxt(questionNum+phase+'Q1_ConvW3.out', np.reshape(Q1_ConvW3.eval(session=sess),[-1]))
    np.savetxt(questionNum+phase+'Q1_ConvB3.out', Q1_ConvB3.eval(session=sess))
    np.savetxt(questionNum+phase+'Q1_FC_w1.out', Q1_FC_w1.eval(session=sess))
    np.savetxt(questionNum+phase+'Q1_FC_b1.out', Q1_FC_b1.eval(session=sess))
    np.savetxt(questionNum+phase+'Q1_FC_w2.out', Q1_FC_w2.eval(session=sess))
    np.savetxt(questionNum+phase+'Q1_FC_b2.out', Q1_FC_b2.eval(session=sess))
    
    np.savetxt(questionNum+phase+'Q2_ConvW1.out', np.reshape(Q1_ConvW1.eval(session = sess),[-1]))
    np.savetxt(questionNum+phase+'Q2_ConvB1.out', Q2_ConvB1.eval(session=sess))
    np.savetxt(questionNum+phase+'Q2_ConvW2.out', np.reshape(Q1_ConvW2.eval(session = sess),[-1]))
    np.savetxt(questionNum+phase+'Q2_ConvB2.out', Q2_ConvB2.eval(session=sess))
    np.savetxt(questionNum+phase+'Q2_ConvW3.out', np.reshape(Q1_ConvW3.eval(session=sess),[-1]))
    np.savetxt(questionNum+phase+'Q2_ConvB3.out', Q2_ConvB3.eval(session=sess))
    np.savetxt(questionNum+phase+'Q2_FC_w1.out', Q2_FC_w1.eval(session=sess))
    np.savetxt(questionNum+phase+'Q2_FC_b1.out', Q2_FC_b1.eval(session=sess))
    np.savetxt(questionNum+phase+'Q2_FC_w2.out', Q2_FC_w2.eval(session=sess))
    np.savetxt(questionNum+phase+'Q2_FC_b2.out', Q2_FC_b2.eval(session=sess))
    
    
    


def ImagePreprocess(input_image):
    #input image is either uint8 or I make it uint8 to be suitable stored on
    #memory
    #return: processed image as uint8
    beforeProcess = np.asarray(input_image, dtype = np.uint8)
    temp_image = cv2.cvtColor(beforeProcess, cv2.COLOR_RGB2GRAY)
    resizedImage = cv2.resize(temp_image,(84, 84), interpolation=cv2.INTER_AREA)
    return resizedImage

def ExecuteOneStep(env, sess, lastStateProcessed, lastState, output_Q, p):
    #Evaluate the current online Q function and then pick up one action
    #using the linear epsilon greedy policy to execute in the emulator
    #observe the next state and reward and then make a sample of that
    #RETURN: one sample
    q_value = output_Q.eval(feed_dict = {x1:lastStateProcessed,x2:lastStateProcessed},session = sess)
    action = p.select_action(q_value)
    frame5, reward5, is_terminal5, info = env.step(action)
    frame6, reward6, is_terminal6, info = env.step(action)
    frame7, reward7, is_terminal7, info = env.step(action)
    nextState = [np.copy(frame5), np.copy(frame6), np.copy(frame7)]
    terminal = max([is_terminal5, is_terminal6, is_terminal7])
    reward = [reward5, reward6, reward7]
    tempSample = Sample(lastState, action, reward, nextState, terminal)
        
    return copy.deepcopy(tempSample)
    
def DoubleQEvaluateAction(next, Conv_W1, Conv_B1, Conv_W2, Conv_B2, Conv_W3, Conv_B3, FC_w1, FC_b1, FC_w2, FC_b2, out_dim1, out_dim2, out_dim3, output_num_final):
    # Use either Q1 or Q2 to evaluate the next state and then pass the action
    # to another network to get the value evaluation.
    # RETURN:
    # this function return {Q1(s`,a)} or {Q2(s`,a)} depending
    # on the input parameters
    conv1 = tf.nn.conv2d(next, Conv_W1, [1,1,4,4], padding='VALID', data_format = 'NCHW')
    out1 = tf.nn.bias_add(conv1, Conv_B1, data_format = 'NCHW')
    conv_out1 = tf.nn.relu(out1)
    conv2 = tf.nn.conv2d(conv_out1, Conv_W2, [1,1,2,2], padding='VALID', data_format = 'NCHW')
    out2 = tf.nn.bias_add(conv2, Conv_B2, data_format = 'NCHW')
    conv_out2 = tf.nn.relu(out2)
    conv3 = tf.nn.conv2d(conv_out2, Conv_W3, [1,1,1,1], padding='VALID', data_format = 'NCHW')
    out3 = tf.nn.bias_add(conv3, Conv_B3, data_format = 'NCHW')
    conv_out3 = tf.nn.relu(out3)
    conv_out3_flat = tf.reshape(conv_out3, [-1,7*7*64])
    FC_out1 = tf.nn.relu(tf.matmul(conv_out3_flat, FC_w1) + FC_b1)
    FC_out2 = tf.nn.relu(tf.matmul(FC_out1, FC_w2) + FC_b2)
    return FC_out2
    


def FindIndexForDoubleQ(batch_size, index1argmax, index2argmax,feed, sess):
    #inder to make the tf.gather_nd pick up the right value from Q
    # we need to create this function
    # return index1feed and index2feed
    argmax1 = index1argmax.eval(feed_dict=feed,session=sess)
    argmax2 = index2argmax.eval(feed_dict=feed,session=sess)
    index1feed = []
    index2feed = []
    for i in range(batch_size):
        index1feed.append([i,argmax1[i]])
        index2feed.append([i,argmax2[i]])
        
    return index1feed,index2feed

#===========MAIN FUNCTION STARTS FROM HERE====================
#==========HYPERPARAMETER SET UP==============================
gamma = 0.99 #discount factor
alpha = 0.0005 #Learning Rate
num_iteration = 300000
#ATTENTION TO THE FOLLOWING VARIABLE:
rewardOneEpisode = 0 # sum the reward obtained from 1 episode
Max_TimeStep = 10000#this is T described in the reference 2
#Make Max_TimeStep to be small so that I can debug on my PC
#Change that to big value when upload to the super computer!!!
#num_update_target = 100 #target update after this number of iterations
#Again, make it small first to debug locally and then make big when submit!
update_counter = 0 #used for debug only, will not affect the algorithm
batch_size = 32
env = gym.make('SpaceInvaders-v0')
#if we use SpaceInvaders as suggested by the paper, we should use 3,
#while all the other games are required that we use 4
num_frame_skip = 3
#==========Neural Nets Parameters are here=================
output_num_final = env.action_space.n
LinearPolicy = policy.LinearDecayGreedyEpsilonPolicy(output_num_final, 1, 0.05, 200000)
out_dim1 = 32 #number of filters for convolutional layer1
out_dim2 = 64 #number of filters for convolutional layer2
out_dim3 = 64 #number of filters for convolutional layer3

experience = Experience()


#============BUILD NETWORK===========================
# the data fomat used here is "NCHW"!!!
#define session to run:
sess = tf.Session()
#define placeholders for state, action, reward, nextstate, terminal
index1 = tf.placeholder(tf.int32, shape = [None, 2], name = 'index1')
index2 = tf.placeholder(tf.int32, shape = [None, 2], name = 'index2')
action = tf.placeholder(tf.int32, shape = [None, 2], name = 'action')
terminal = tf.placeholder(tf.float32, shape = [None, 1], name = 'terminal')
r = tf.placeholder(tf.float32, shape = [None, 1], name = 'r')
x1 = tf.placeholder(tf.float32, shape = [None, num_frame_skip, Sample.height, Sample.width], name = 'x1')
next_x1 = tf.placeholder(tf.float32, shape = [None, num_frame_skip, Sample.height, Sample.width], name = 'next_x1')
x2 = tf.placeholder(tf.float32, shape = [None, num_frame_skip, Sample.height, Sample.width], name = 'x2')
next_x2 = tf.placeholder(tf.float32, shape = [None, num_frame_skip, Sample.height, Sample.width], name = 'next_x2')
#NOTE: Since this file implements the double Q network, I will construct the 
#trainning step for the two loss functions sperately and then flip the coin to 
#decide which loss to be optimized for each step. Then x1 denotes the states 
#input for Q1 network and x2 denotes the states input for Q2 while next_x1 and 
#next_x2 denote the next_state input for the 2 network respectively.


#=========BUILD ONLINE NETWORK-ONE========================================
#========build 1st convolutional layer=========
Q1_ConvW1 = tf.Variable(tf.truncated_normal([8,8,Sample.historyLen, out_dim1],stddev = 0.1), name = 'Q1_ConvW1')
Q1_ConvB1 = tf.Variable(tf.truncated_normal([out_dim1],stddev=0.1), name = "Q1_ConvB1")
Q1_conv1 = tf.nn.conv2d(x1, Q1_ConvW1, [1,1,4,4], padding='VALID', data_format='NCHW')
Q1_out1 = tf.nn.bias_add(Q1_conv1,Q1_ConvB1,data_format='NCHW')
Q1_conv_out1 = tf.nn.relu(Q1_out1)
#========build 2nd convolutional layer=========
Q1_ConvW2 = tf.Variable(tf.truncated_normal([4,4,out_dim1, out_dim2],stddev = 0.1), name = 'Q1_ConvW2')
Q1_ConvB2 = tf.Variable(tf.truncated_normal([out_dim2],stddev=0.1), name = 'Q1_ConvB2')
Q1_conv2 = tf.nn.conv2d(Q1_conv_out1,Q1_ConvW2,[1,1,2,2],padding='VALID', data_format='NCHW')
Q1_out2 = tf.nn.bias_add(Q1_conv2,Q1_ConvB2,data_format='NCHW')
Q1_conv_out2 = tf.nn.relu(Q1_out2)
#========build 3rd convolutional layer=========
Q1_ConvW3 = tf.Variable(tf.truncated_normal([3,3,out_dim2, out_dim3], stddev=0.1), name = 'Q1_ConvW3')
Q1_ConvB3 = tf.Variable(tf.truncated_normal([out_dim3],stddev=0.1), name = 'Q1_ConvB3')
Q1_conv3 = tf.nn.conv2d(Q1_conv_out2,Q1_ConvW3,[1,1,1,1],padding='VALID', data_format='NCHW')
Q1_out3 = tf.nn.bias_add(Q1_conv3,Q1_ConvB3,data_format='NCHW')
Q1_conv_out3 = tf.nn.relu(Q1_out3)
Q1_conv_out3_flat = tf.reshape(Q1_conv_out3,[-1, 7*7*64])
#========build 1st fully connected layer===========
Q1_FC_w1 = tf.Variable(tf.truncated_normal([7*7*64, 512],stddev=0.1),name='Q1_FC_w1')
Q1_FC_b1 = tf.Variable(tf.truncated_normal([512],stddev=0.1),name='Q1_FC_b1')
Q1_FC_out1 = tf.nn.relu(tf.matmul(Q1_conv_out3_flat, Q1_FC_w1) + Q1_FC_b1)
#========build final fully connected layer=========
Q1_FC_w2 = tf.Variable(tf.truncated_normal([512, output_num_final],stddev=0.1), name = 'Q1_FC_w2')
Q1_FC_b2 = tf.Variable(tf.truncated_normal([output_num_final],stddev=0.1), name = 'Q1_FC_b2')
Q1_FC_out2 = tf.matmul(Q1_FC_out1, Q1_FC_w2) + Q1_FC_b2

#============BUILD ONLINE NETWORK-TWO==================================
#========build 1st convolutional layer=========
Q2_ConvW1 = tf.Variable(tf.truncated_normal([8,8,Sample.historyLen, out_dim1],stddev = 0.1), name = 'Q2_ConvW1')
Q2_ConvB1 = tf.Variable(tf.truncated_normal([out_dim1],stddev=0.1), name = "Q2_ConvB1")
Q2_conv1 = tf.nn.conv2d(x2, Q2_ConvW1, [1,1,4,4], padding='VALID', data_format='NCHW')
Q2_out1 = tf.nn.bias_add(Q2_conv1,Q2_ConvB1,data_format='NCHW')
Q2_conv_out1 = tf.nn.relu(Q2_out1)
#========build 2nd convolutional layer=========
Q2_ConvW2 = tf.Variable(tf.truncated_normal([4,4,out_dim1, out_dim2],stddev = 0.1), name = 'Q2_ConvW2')
Q2_ConvB2 = tf.Variable(tf.truncated_normal([out_dim2],stddev=0.1), name = 'Q1_ConvB2')
Q2_conv2 = tf.nn.conv2d(Q2_conv_out1,Q2_ConvW2,[1,1,2,2],padding='VALID', data_format='NCHW')
Q2_out2 = tf.nn.bias_add(Q2_conv2,Q2_ConvB2,data_format='NCHW')
Q2_conv_out2 = tf.nn.relu(Q2_out2)
#========build 3rd convolutional layer=========
Q2_ConvW3 = tf.Variable(tf.truncated_normal([3,3,out_dim2, out_dim3], stddev=0.1), name = 'Q2_ConvW3')
Q2_ConvB3 = tf.Variable(tf.truncated_normal([out_dim3],stddev=0.1), name = 'Q2_ConvB3')
Q2_conv3 = tf.nn.conv2d(Q2_conv_out2,Q2_ConvW3,[1,1,1,1],padding='VALID', data_format='NCHW')
Q2_out3 = tf.nn.bias_add(Q2_conv3,Q2_ConvB3,data_format='NCHW')
Q2_conv_out3 = tf.nn.relu(Q2_out3)
Q2_conv_out3_flat = tf.reshape(Q2_conv_out3,[-1, 7*7*64])
#========build 1st fully connected layer===========
Q2_FC_w1 = tf.Variable(tf.truncated_normal([7*7*64, 512],stddev=0.1),name='Q2_FC_w1')
Q2_FC_b1 = tf.Variable(tf.truncated_normal([512],stddev=0.1),name='Q1_FC_b1')
Q2_FC_out1 = tf.nn.relu(tf.matmul(Q2_conv_out3_flat, Q2_FC_w1) + Q2_FC_b1)
#========build final fully connected layer=========
Q2_FC_w2 = tf.Variable(tf.truncated_normal([512, output_num_final],stddev=0.1), name = 'Q2_FC_w2')
Q2_FC_b2 = tf.Variable(tf.truncated_normal([output_num_final],stddev=0.1), name = 'Q2_FC_b2')
Q2_FC_out2 = tf.matmul(Q2_FC_out1, Q2_FC_w2) + Q2_FC_b2


#=========CONSTRUCT Q1 and Q2 NETWORK TAKING NEXT_STATE AS INPUT===============

output_Q1_state = Q1_FC_out2
output_Q2_state = Q2_FC_out2

output_Q = tf.add(output_Q1_state, output_Q2_state)

actionEval_by_Q1 = DoubleQEvaluateAction(next_x1, Q1_ConvW1, Q1_ConvB1, Q1_ConvW2, Q1_ConvB2, Q1_ConvW3, Q1_ConvB3, Q1_FC_w1, Q1_FC_b1, Q1_FC_w2, Q1_FC_b2, out_dim1, out_dim2, out_dim3, output_num_final)
actionEval_by_Q2 = DoubleQEvaluateAction(next_x2, Q2_ConvW1, Q2_ConvB1, Q2_ConvW2, Q2_ConvB2, Q2_ConvW3, Q2_ConvB3, Q2_FC_w1, Q2_FC_b1, Q2_FC_w2, Q2_FC_b2, out_dim1, out_dim2, out_dim3, output_num_final)
index1argmax = tf.argmax(actionEval_by_Q1, axis=1)
y_true1 = r + gamma*tf.gather_nd(actionEval_by_Q2, index1) * (1-terminal)
y_pred1 = tf.gather_nd(output_Q1_state, action)

index2argmax = tf.argmax(actionEval_by_Q2, axis=1)
y_true2 = r + gamma*tf.gather_nd(actionEval_by_Q1, index2) * (1-terminal)
y_pred2 = tf.gather_nd(output_Q2_state, action)


#create the loss function and set up the training step:
#samples and form a batch to calculate the loss
loss1 = tf.reduce_mean(objectives.huber_loss(y_true1, y_pred1))
loss2 = tf.reduce_mean(objectives.huber_loss(y_true2, y_pred2))

train_step1 = tf.train.AdamOptimizer(alpha).minimize(loss1)
train_step2 = tf.train.AdamOptimizer(alpha).minimize(loss2)
sess.run(tf.global_variables_initializer())
#==================DOING THE TRAINING LOOP=========================
#NOTE: if current number of samples is less than the batch size I will create t
#the batch first by performing random action to get enough samples.
update_counter = 0
Q6Saver(Q1_ConvW1, Q1_ConvB1, Q1_ConvW2, Q1_ConvB2, Q1_ConvW3, Q1_ConvB3, Q1_FC_w1, Q1_FC_b1, Q1_FC_w2, Q1_FC_b2, Q2_ConvW1, Q2_ConvB1, Q2_ConvW2, Q2_ConvB2, Q2_ConvW3, Q2_ConvB3, Q2_FC_w1, Q2_FC_b1, Q2_FC_w2, Q2_FC_b2, 'P0', sess)
while update_counter < num_iteration:
    env.reset()
    rewardOneEpisode = 0
    for j in range(batch_size, Max_TimeStep):
        stateBatch, actionBatch, rewardBatch, nextStateBatch, terminalBatch = \
        experience.getRandomBatch(env, sess, batch_size, output_Q,
        LinearPolicy)
        if j == batch_size:
            update_counter = update_counter + batch_size
            rewardOneEpisode = np.sum(rewardBatch)
        else:
            update_counter = update_counter + 1
            rewardOneEpisode = rewardOneEpisode + recentReward
        feed = {x1:stateBatch, x2:stateBatch, next_x1:nextStateBatch, next_x2:nextStateBatch, action:actionBatch, r:rewardBatch, terminal: terminalBatch}
        index1feed,index2feed = FindIndexForDoubleQ(batch_size, index1argmax, index2argmax,feed, sess)
        feed = {x1:stateBatch, x2:stateBatch, next_x1:nextStateBatch, next_x2:nextStateBatch, action:actionBatch, r:rewardBatch, terminal: terminalBatch, index1:index1feed, index2:index2feed}
        if np.random.random() > 0.5:
            train_step1.run(feed_dict = feed, session = sess)
        else:
            train_step2.run(feed_dict = feed, session = sess)
        lastStateProcessed, lastState = experience.getLastState()
        sample = ExecuteOneStep(env, sess, lastStateProcessed, lastState, output_Q, LinearPolicy)
        experience.addSample(sample)
        recentReward = sample.getReward()[0]
        if update_counter == int(num_iteration/3 * 1):
            Q6Saver(Q1_ConvW1, Q1_ConvB1, Q1_ConvW2, Q1_ConvB2, Q1_ConvW3, Q1_ConvB3, Q1_FC_w1, Q1_FC_b1, Q1_FC_w2, Q1_FC_b2, Q2_ConvW1, Q2_ConvB1, Q2_ConvW2, Q2_ConvB2, Q2_ConvW3, Q2_ConvB3, Q2_FC_w1, Q2_FC_b1, Q2_FC_w2, Q2_FC_b2, 'P1', sess)
        elif update_counter == int(num_iteration/3 * 2):
            Q6Saver(Q1_ConvW1, Q1_ConvB1, Q1_ConvW2, Q1_ConvB2, Q1_ConvW3, Q1_ConvB3, Q1_FC_w1, Q1_FC_b1, Q1_FC_w2, Q1_FC_b2, Q2_ConvW1, Q2_ConvB1, Q2_ConvW2, Q2_ConvB2, Q2_ConvW3, Q2_ConvB3, Q2_FC_w1, Q2_FC_b1, Q2_FC_w2, Q2_FC_b2, 'P2', sess)
        if bool(sample.terminal):
            break
    

    print('Complete:')
    print(update_counter/num_iteration * 100)
    print("Last episode total reward was:")
    print(rewardOneEpisode)
    print("Experience lenght is;")
    print(len(experience.SamplePool))
    
Q6Saver(Q1_ConvW1, Q1_ConvB1, Q1_ConvW2, Q1_ConvB2, Q1_ConvW3, Q1_ConvB3, Q1_FC_w1, Q1_FC_b1, Q1_FC_w2, Q1_FC_b2, Q2_ConvW1, Q2_ConvB1, Q2_ConvW2, Q2_ConvB2, Q2_ConvW3, Q2_ConvB3, Q2_FC_w1, Q2_FC_b1, Q2_FC_w2, Q2_FC_b2, 'P3', sess)    