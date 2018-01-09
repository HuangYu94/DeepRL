import gym
import tensorflow as tf
import numpy as np
import random
import policy
import objectives
import copy
import cv2
#This file implements the problem3 linear Q network with experience replay and
# and target fixing to the extent possible as describled in reference 1 and 2
#===================utility classes are defined as follows===================
class Sample:
    #This class store one sample of the tuple (s,r,a,s`,terminal)
    def __init__(self, state_in, action_in, reward_in, nextstate_in, is_terminal):
        #input states, action, reward are all lists of 3 frames since I use the
        #no frame skipping version of SpaceInvaders
        self.state = state_in
        self.nextstate = nextstate_in
        self.action = action_in
        self.terminal = is_terminal
        reward = sum(reward_in)
        self.reward=reward

    def getProcessedState(self):
        #Flatten the current state and next state as [4*84*84] respectively:
        #We convert the uint8 data type to float32 type to let the neural
        #network better process the image
        #return: processed image as float32
        for i, oneframe in enumerate(self.state):
            tempFrame = ImagePreprocess(oneframe)
            flatten = tempFrame.flatten()
            if i == 0:
                processedState = copy.deepcopy(flatten)
            else:
                processedState = np.append(processedState, flatten)
        temp = np.asarray(processedState, dtype=np.float32)
        temp = temp.reshape([1,-1])
        return temp
    
    def getProcessedNextState(self):
        #similiar to the previous function
        #this just return the processed next state stored in the sample
        for i, oneframe in enumerate(self.nextstate):
            tempFrame = ImagePreprocess(oneframe)
            flatten = tempFrame.flatten()
            if i == 0:
                processedState = copy.deepcopy(flatten)
            else:
                processedState = np.append(processedState, flatten)
        temp = np.asarray(processedState, dtype=np.float32)
        temp = temp.reshape([1,-1])
        return temp

        
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
def Q3Saver(online_weight, online_bias, trainPhase, sess):
    #this function will save the trained parameters 4 times during the 
    #trainning process
    #all the weights will be saved by numpy.savetxt()
    #The naming convention I use is format like Q2_P0_weights.out for example
    #NO RETURN
    weights = online_weight.eval(session = sess)
    bias = online_bias.eval(session = sess)
    questionString = 'Q3'
    weightName = 'weights.out'
    biasName = 'bias.out'
    trainPhase = '_'+trainPhase+'_'
    weightsFile = questionString+trainPhase + weightName
    np.savetxt(weightsFile,weights)
    biasFile = questionString+trainPhase+biasName
    np.savetxt(biasFile,bias)

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
Max_TimeStep = 10000#this is T described in the reference 2
#Make Max_TimeStep to be small so that I can debug on my PC
#Change that to big value when upload to the super computer
num_update_target = 10000 #target update after this number of iterations
#Again, make it small first to debug locally and then make big when submit!
batch_size = 32
env = gym.make('SpaceInvaders-v0')
#if we use SpaceInvaders as suggested by the paper, we should use 3,
#while all the other games are required that we use 4
num_frame_skip = 3
output_num = env.action_space.n
LinearPolicy = policy.LinearDecayGreedyEpsilonPolicy(output_num, 1, 0.05, 100000)
experience = Experience()


#============BUILD NETWORK===========================
#define session to run:
sess = tf.Session()
#define placeholders for state, action, reward, nextstate, terminal
action = tf.placeholder(tf.int32, shape = [None,2], name = 'action')
terminal = tf.placeholder(tf.float32, shape = [None, 1], name = 'terminal')
r = tf.placeholder(tf.float32, shape = [None, 1], name = 'r')
x = tf.placeholder(tf.float32, shape = [None, num_frame_skip*84*84], name = 'x')
next_x = tf.placeholder(tf.float32, shape = [None, num_frame_skip*84*84], name = 'next_x')
online_weight = tf.Variable(tf.truncated_normal([num_frame_skip*84*84, output_num],stddev = 0.1), name = 'online_weight')
target_weight = tf.placeholder(dtype = tf.float32, shape = [num_frame_skip*84*84, output_num], name = 'target_weight')
online_bias = tf.Variable(tf.zeros([output_num]),dtype = tf.float32, name='online_bias')
target_bias = tf.placeholder(dtype = tf.float32, shape = [output_num], name='target_bias')

output_Q = tf.matmul(x, online_weight) + online_bias
target_Q = tf.matmul(next_x, target_weight) + target_bias

y_true = r + gamma * tf.reduce_max(target_Q) * (1-terminal)
y_pred = tf.gather_nd(output_Q, action)

#create the loss function and set up the training step:
#samples and form a batch to calculate the loss
loss = tf.reduce_mean(objectives.huber_loss(y_true, y_pred))

train_step = tf.train.AdamOptimizer(alpha).minimize(loss)
sess.run(tf.global_variables_initializer())
#==================DOING THE TRAINING LOOP=========================
#NOTE: if current number of samples is less than the batch size I will create t
#the batch first by performing random action to get enough samples.
Q3Saver(online_weight, online_bias, 'P0', sess)
update_counter = 0 #used for target update
while update_counter < num_iteration:
    env.reset()
    rewardOneEpisode = 0 # collect reward obtained from one episode.
    for j in range(batch_size, Max_TimeStep):
        if update_counter % num_update_target == 0 or j==batch_size:
            target_w = online_weight.eval(session = sess)
            target_b = online_bias.eval(session = sess)
        stateBatch, actionBatch, rewardBatch, nextStateBatch, terminalBatch = \
        experience.getRandomBatch(env, sess, batch_size, output_Q,
        LinearPolicy)
        feed = {x:stateBatch, action:actionBatch, r:rewardBatch, next_x:
            nextStateBatch, terminal: terminalBatch, target_weight:target_w,
            target_bias:target_b}
        train_step.run(feed_dict = feed,session = sess)
        if j == batch_size:
            rewardOneEpisode = np.sum(rewardBatch)
            update_counter = update_counter + batch_size
        else:
            rewardOneEpisode += recentReward
            update_counter = update_counter + 1
        lastStateProcessed, lastState = experience.getLastState()
        sample = ExecuteOneStep(env, sess, lastStateProcessed, lastState, output_Q, LinearPolicy)
        recentReward = sample.getReward()[0]
        experience.addSample(sample)
        if update_counter == int(num_iteration/3 * 1):
            Q3Saver(online_weight, online_bias, 'P1', sess)
        elif update_counter == int(num_iteration/3 * 2):
            Q3Saver(online_weight, online_bias, 'P2', sess)
        if bool(sample.terminal):
            break
        

        
    print('Complete:')
    print(update_counter/num_iteration * 100)
    print("Current total reward is:")
    print(rewardOneEpisode)
    print("Experience lenght is;")
    print(len(experience.SamplePool))
    
Q3Saver(online_weight, online_bias, 'P3', sess)
    