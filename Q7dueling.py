import gym
import tensorflow as tf
import numpy as np
import random
import policy
import objectives
import copy
import cv2
# This file implements the problem5 deep Q network with experience replay and
# and target fixing to the extent possible as describled in reference 1 and 2
# I will try to make the parameter setting similiar to reference 1 and 2 to the 
# extent possible!
# I noticed that reference 1 and 2 have differeence CNN configuration and in 
# this file I implements the configuration introduced in ref 2 since it seems 
# perform better and more complecated!
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
                q_value = output_Q.eval(feed_dict = {x:last_state},
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
def Q7Saver(ConvW1, ConvB1, ConvW2, ConvB2, ConvW3, ConvB3, Value_FC_w1, Value_FC_b1, Advatange_FC_w1, Advantage_FC_b1, Value_FC_w2, Value_FC_b2, Advantage_FC_w2, Advantage_FC_b2, trainPhase, sess):
    # same as before save the weights for different trainning phase.
    # Note that convolution weights are all flattened to be stored
    questionNum = 'Q7'
    phase = '_' + trainPhase + '_'
    np.savetxt(questionNum+phase+'ConvW1.out', np.reshape(ConvW1.eval(session = sess),[-1]))
    np.savetxt(questionNum+phase+'ConvB1.out', ConvB1.eval(session = sess))
    np.savetxt(questionNum+phase+'ConvW2.out', np.reshape(ConvW2.eval(session = sess),[-1]))
    np.savetxt(questionNum+phase+'ConvB2.out', ConvB2.eval(session = sess))
    np.savetxt(questionNum+phase+'ConvW3.out', np.reshape(ConvW3.eval(session = sess),[-1]))
    np.savetxt(questionNum+phase+'ConvB3.out', ConvB3.eval(session = sess))
    np.savetxt(questionNum+phase+'Value_FC_w1.out', Value_FC_w1.eval(session = sess))
    np.savetxt(questionNum+phase+'Value_FC_b1.out', Value_FC_b1.eval(session = sess))
    np.savetxt(questionNum+phase+'Advantage_FC_w1.out', Advantage_FC_w1.eval(session = sess))
    np.savetxt(questionNum+phase+'Advantage_FC_b1.out', Advantage_FC_b1.eval(session = sess))
    np.savetxt(questionNum+phase+'Value_FC_w2.out', Value_FC_w2.eval(session = sess))
    np.savetxt(questionNum+phase+'Value_FC_b2.out', Value_FC_b2.eval(session = sess))
    np.savetxt(questionNum+phase+'Advantage_FC_w2.out', Advantage_FC_w2.eval(session = sess))
    np.savetxt(questionNum+phase+'Advantage_FC_b2.out', Advantage_FC_b2.eval(session = sess))


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
    q_value = output_Q.eval(feed_dict = {x:lastStateProcessed},session = sess)
    action = p.select_action(q_value)
    frame5, reward5, is_terminal5, info = env.step(action)
    frame6, reward6, is_terminal6, info = env.step(action)
    frame7, reward7, is_terminal7, info = env.step(action)
    nextState = [np.copy(frame5), np.copy(frame6), np.copy(frame7)]
    terminal = max([is_terminal5, is_terminal6, is_terminal7])
    reward = [reward5, reward6, reward7]
    tempSample = Sample(lastState, action, reward, nextState, terminal)
        
    return copy.deepcopy(tempSample)
    
    
    

#===========MAIN FUNCTION STARTS FROM HERE====================
#==========HYPERPARAMETER SET UP==============================
gamma = 0.99 #discount factor
alpha = 0.001 #Learning Rate
num_iteration = 300000
#ATTENTION TO THE FOLLOWING VARIABLE:
rewardOneEpisode = 0 # sum the reward obtained from 1 episode
Max_TimeStep = 10000#this is T described in the reference 2
#Make Max_TimeStep to be small so that I can debug on my PC
#Change that to big value when upload to the super computer!!!
num_update_target = 10000 #target update after this number of iterations
#Again, make it small first to debug locally and then make big when submit!
update_counter = 0 #used for debug only, will not affect the algorithm
batch_size = 32
env = gym.make('SpaceInvaders-v0')
#if we use SpaceInvaders as suggested by the paper, we should use 3,
#while all the other games are required that we use 4
num_frame_skip = 3
#==========Neural Nets Parameters are here=================
output_num_final = env.action_space.n
LinearPolicy = policy.LinearDecayGreedyEpsilonPolicy(output_num_final, 1, 0.05, 100000)
out_dim1 = 32 #number of filters for convolutional layer1
out_dim2 = 64 #number of filters for convolutional layer2
out_dim3 = 64 #number of filters for convolutional layer3

experience = Experience()


#============BUILD NETWORK===========================
# the data fomat used here is "NCHW"!!!
#define session to run:
sess = tf.Session()
#define placeholders for state, action, reward, nextstate, terminal
action = tf.placeholder(tf.int32, shape = [None,2], name = 'action')
terminal = tf.placeholder(tf.float32, shape = [None, 1], name = 'terminal')
r = tf.placeholder(tf.float32, shape = [None, 1], name = 'r')
x = tf.placeholder(tf.float32, shape = [None, num_frame_skip, Sample.height, Sample.width], name = 'x')
next_x = tf.placeholder(tf.float32, shape = [None, num_frame_skip, Sample.height, Sample.width], name = 'next_x')
#=========BUILD ONLINE NETWORK========================
#========build 1st convolutional layer=========
online_ConvW1 = tf.Variable(tf.truncated_normal([8,8,Sample.historyLen, out_dim1],stddev = 0.1), name = 'online_ConvW1')
online_ConvB1 = tf.Variable(tf.truncated_normal([out_dim1],stddev=0.1), name = "online_ConvB1")
conv1 = tf.nn.conv2d(x, online_ConvW1, [1,1,4,4], padding='VALID', data_format='NCHW')
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
#========build 1st fully connected layer of value function===========
Value_FC_w1 = tf.Variable(tf.truncated_normal([7*7*64, 512],stddev=0.1),name='Value_FC_w1')
Value_FC_b1 = tf.Variable(tf.truncated_normal([512],stddev=0.1),name='Value_FC_b1')
Value_FC_out1 = tf.nn.relu(tf.matmul(conv_out3_flat, Value_FC_w1) + Value_FC_b1)
#========build final fully connected layer of value function=========
Value_FC_w2 = tf.Variable(tf.truncated_normal([512, 1],stddev=0.1), name = 'Value_FC_w2')
Value_FC_b2 = tf.Variable(tf.truncated_normal([1],stddev=0.1), name = 'Value_FC_b2')
Value_FC_out2 = tf.matmul(Value_FC_out1, Value_FC_w2) + Value_FC_b2
#========build 1st fully connected layer of advantage function=======
Advantage_FC_w1 = tf.Variable(tf.truncated_normal([7*7*64,  512],stddev=0.1),name='Advantage_FC_w1')
Advantage_FC_b1 = tf.Variable(tf.truncated_normal([512],stddev=0.1),name='Advantage_FC_b1')
Advantage_FC_out1 = tf.matmul(conv_out3_flat, Advantage_FC_w1) + Advantage_FC_b1
#========build final fully connected layer of advantage function=====
Advantage_FC_w2 = tf.Variable(tf.truncated_normal([512, output_num_final],stddev=0.1),name='Advantage_FC_w2')
Advantage_FC_b2 = tf.Variable(tf.truncated_normal([output_num_final],stddev=0.1),name='Advantage_FC_b2')
Advantage_FC_out2 = tf.matmul(Advantage_FC_out1, Advantage_FC_w2)+Advantage_FC_b2




#============BUILD TARGET NETWORK==================================
#============build 1st convolutional layer==========
target_ConvW1 = tf.placeholder(dtype = tf.float32, shape = [8,8,Sample.historyLen, out_dim1], name = 'target_ConvW1')
target_ConvB1 = tf.placeholder(dtype = tf.float32, shape = [out_dim1], name = 'target_ConvB1')
conv1_target = tf.nn.conv2d(next_x, target_ConvW1, [1,1,4,4], padding='VALID', data_format='NCHW')
out1_target = tf.nn.bias_add(conv1_target,target_ConvB1,data_format='NCHW')
conv_out1_target = tf.nn.relu(out1_target)
#============build 2nd convolutional layer==========
target_ConvW2 = tf.placeholder(dtype = tf.float32, shape = [4,4,out_dim1, out_dim2], name = 'target_ConvW2')
target_ConvB2 = tf.placeholder(dtype = tf.float32, shape = [out_dim2], name = 'target_ConvB2')
conv2_target = tf.nn.conv2d(conv_out1_target,target_ConvW2,[1,1,2,2],padding='VALID', data_format='NCHW')
out2_target = tf.nn.bias_add(conv2_target,target_ConvB2,data_format='NCHW')
conv_out2_target = tf.nn.relu(out2_target)
#========build 3rd convolutional layer=========
target_ConvW3 = tf.placeholder(dtype = tf.float32, shape = [3,3,out_dim2, out_dim3], name = 'target_ConvW3')
target_ConvB3 = tf.placeholder(dtype = tf.float32, shape = [out_dim3], name = 'online_ConvB3')
conv3_target = tf.nn.conv2d(conv_out2_target,target_ConvW3,[1,1,1,1],padding='VALID', data_format='NCHW')
out3_target = tf.nn.bias_add(conv3_target,target_ConvB3,data_format='NCHW')
conv_out3_target = tf.nn.relu(out3_target)
conv_out3_target_flat = tf.reshape(conv_out3_target,[-1, 7*7*64])
#========build 1st fully connected layer for target value network===========
V_target_FC_w1 = tf.placeholder(dtype = tf.float32, shape = [7*7*64, 512], name='V_target_FC_w1')
V_target_FC_b1 = tf.placeholder(dtype = tf.float32, shape = [512], name='V_target_FC_b1')
V_FC_out1_target = tf.nn.relu(tf.matmul(conv_out3_target_flat, V_target_FC_w1) + V_target_FC_b1)
#========build final fully connected layer for target value network=========
V_target_FC_w2 = tf.placeholder(dtype = tf.float32, shape = [512, 1], name = 'V_target_FC_w2')
V_target_FC_b2 = tf.placeholder(dtype = tf.float32, shape = [1], name = 'V_target_FC_b2')
V_FC_out2_target = tf.matmul(V_FC_out1_target, V_target_FC_w2) + V_target_FC_b2
#========build 1st fully connected layer for target advantage network===========
A_target_FC_w1 = tf.placeholder(dtype = tf.float32, shape = [7*7*64, 512], name='A_target_FC_w1')
A_target_FC_b1 = tf.placeholder(dtype = tf.float32, shape = [512], name='A_target_FC_b1')
A_FC_out1_target = tf.nn.relu(tf.matmul(conv_out3_target_flat, A_target_FC_w1) + A_target_FC_b1)
#========build final fully connected layer for target value network========
A_target_FC_w2 = tf.placeholder(dtype = tf.float32, shape = [512, output_num_final], name='A_target_FC_w2')
A_target_FC_b2 = tf.placeholder(dtype = tf.float32, shape = [output_num_final], name='A_target_FC_b2')
A_FC_out2_target = tf.nn.relu(tf.matmul(A_FC_out1_target, A_target_FC_w2) + A_target_FC_b2)

#=======building loss function=================================

target_Q = V_FC_out2_target + tf.reduce_max(A_FC_out2_target) - tf.reduce_mean(A_FC_out2_target)
online_Q = Value_FC_out2 + tf.gather_nd(Advantage_FC_out2, action) - tf.reduce_mean(Advantage_FC_out2)
online_Qeval = Advantage_FC_out2

y_true = r + gamma * target_Q * (1-terminal)
y_pred = online_Q

#create the loss function and set up the training step:
#samples and form a batch to calculate the loss
loss = tf.reduce_mean(objectives.huber_loss(y_true, y_pred))

train_step = tf.train.AdamOptimizer(alpha).minimize(loss)
sess.run(tf.global_variables_initializer())
#==================DOING THE TRAINING LOOP=========================
#NOTE: if current number of samples is less than the batch size I will create t
#the batch first by performing random action to get enough samples.
update_counter = 0
Q7Saver(online_ConvW1, online_ConvB1, online_ConvW2, online_ConvB2, online_ConvW3, online_ConvB3, Value_FC_w1, Value_FC_b1, Advantage_FC_w1, Advantage_FC_b1, Value_FC_w2, Value_FC_b2, Advantage_FC_w2, Advantage_FC_b2, 'P0', sess)
while update_counter < num_iteration:
    env.reset()
    rewardOneEpisode = 0
    for j in range(batch_size, Max_TimeStep):
        if update_counter % num_update_target == 0 or j==batch_size-1:
            #Evaluate Current online weight and update target weight:
            target_ConvW1_feed = online_ConvW1.eval(session = sess)
            target_ConvB1_feed = online_ConvB1.eval(session = sess)
            target_ConvW2_feed = online_ConvW2.eval(session = sess)
            target_ConvB2_feed = online_ConvB2.eval(session = sess)
            target_ConvW3_feed = online_ConvW3.eval(session = sess)
            target_ConvB3_feed = online_ConvB3.eval(session = sess)
            V_target_FC_w1_feed = Value_FC_w1.eval(session = sess)
            V_target_FC_b1_feed = Value_FC_b1.eval(session = sess)
            A_target_FC_w1_feed = Advantage_FC_w1.eval(session = sess)
            A_target_FC_b1_feed = Advantage_FC_b1.eval(session = sess)
            V_target_FC_w2_feed = Value_FC_w2.eval(session = sess)
            V_target_FC_b2_feed = Value_FC_b2.eval(session = sess)
            A_target_FC_w2_feed = Advantage_FC_w2.eval(session = sess)
            A_target_FC_b2_feed = Advantage_FC_b2.eval(session = sess)
        stateBatch, actionBatch, rewardBatch, nextStateBatch, terminalBatch = \
        experience.getRandomBatch(env, sess, batch_size, online_Qeval,
        LinearPolicy)
        if j == batch_size:
            rewardOneEpisode = np.sum(rewardBatch)
            update_counter = update_counter + batch_size
            
        else:
            rewardOneEpisode = rewardOneEpisode + recentReward
            update_counter = update_counter + 1
        feed = {x:stateBatch, action:actionBatch, r:rewardBatch, next_x:
            nextStateBatch, terminal: terminalBatch, target_ConvW1:target_ConvW1_feed, target_ConvB1:target_ConvB1_feed, target_ConvW2:target_ConvW2_feed, target_ConvB2:target_ConvB2_feed, target_ConvW3:target_ConvW3_feed, target_ConvB3:target_ConvB3_feed, V_target_FC_w1: V_target_FC_w1_feed, V_target_FC_b1:V_target_FC_b1_feed, A_target_FC_w1:A_target_FC_w1_feed, A_target_FC_b1:A_target_FC_b1_feed, V_target_FC_w2:V_target_FC_w2_feed, V_target_FC_b2: V_target_FC_b2_feed, A_target_FC_w2:A_target_FC_w2_feed,A_target_FC_b2: A_target_FC_b2_feed}
        train_step.run(feed_dict = feed,session = sess)
        lastStateProcessed, lastState = experience.getLastState()
        sample = ExecuteOneStep(env, sess, lastStateProcessed, lastState, online_Qeval, LinearPolicy)
        experience.addSample(sample)
        recentReward = sample.getReward()[0]
        update_counter = update_counter + 1
        if update_counter == int(num_iteration/3):
            Q7Saver(online_ConvW1, online_ConvB1, online_ConvW2, online_ConvB2, online_ConvW3, online_ConvB3, Value_FC_w1, Value_FC_b1, Advantage_FC_w1, Advantage_FC_b1, Value_FC_w2, Value_FC_b2, Advantage_FC_w2, Advantage_FC_b2, 'P1', sess)
        elif update_counter == int(num_iteration/3 * 2):
            Q7Saver(online_ConvW1, online_ConvB1, online_ConvW2, online_ConvB2, online_ConvW3, online_ConvB3, Value_FC_w1, Value_FC_b1, Advantage_FC_w1, Advantage_FC_b1, Value_FC_w2, Value_FC_b2, Advantage_FC_w2, Advantage_FC_b2, 'P2', sess)
        if bool(sample.terminal):
            break
    
    
    print('Complete:')
    print(update_counter/num_iteration * 100)
    print("Current total reward is:")
    print(rewardOneEpisode)
    print("Experience lenght is;")
    print(len(experience.SamplePool))
    
Q7Saver(online_ConvW1, online_ConvB1, online_ConvW2, online_ConvB2, online_ConvW3, online_ConvB3, Value_FC_w1, Value_FC_b1, Advantage_FC_w1, Advantage_FC_b1, Value_FC_w2, Value_FC_b2, Advantage_FC_w2, Advantage_FC_b2, 'P3', sess)