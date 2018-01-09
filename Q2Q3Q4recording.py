import numpy as np
import cv2
import gym
from gym import wrappers

def imageMatrixParser(observation):
    temp_image = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    resizedImage = cv2.resize(temp_image, (84, 84), interpolation=cv2.INTER_AREA)
    return resizedImage.flatten()

## Q2
#weightData = np.loadtxt('Q2_P0_weights.out')
#biasData = np.loadtxt('Q2_P0_bias.out')
#weightData = np.loadtxt('Q2_P1_weights.out')
#biasData = np.loadtxt('Q2_P1_bias.out')
#weightData = np.loadtxt('Q2_P2_weights.out')
#biasData = np.loadtxt('Q2_P2_bias.out')
#weightData = np.loadtxt('Q2_P3_weights.out')
#biasData = np.loadtxt('Q2_P3_bias.out')
## Q3
#weightData = np.loadtxt('Q3_P0_weights.out')
#biasData = np.loadtxt('Q3_P0_bias.out')
#weightData = np.loadtxt('Q3_P1_weights.out')
#biasData = np.loadtxt('Q3_P1_bias.out')
#weightData = np.loadtxt('Q3_P2_weights.out')
#biasData = np.loadtxt('Q3_P2_bias.out')
#weightData = np.loadtxt('Q3_P3_weights.out')
#biasData = np.loadtxt('Q3_P3_bias.out')
## Q4
# weightData = np.loadtxt('Q4_P0_weights1.out')
# biasData = np.loadtxt('Q4_P0_bias1.out')
# weightData2nd = np.loadtxt('Q4_P0_weights2.out')
# biasData2nd = np.loadtxt('Q4_P0_bias2.out')
weightData = np.loadtxt('Q4_P1_weights1.out')
biasData = np.loadtxt('Q4_P1_bias1.out')
weightData2nd = np.loadtxt('Q4_P1_weights2.out')
biasData2nd = np.loadtxt('Q4_P1_bias2.out')
#weightData = np.loadtxt('Q4_P2_weights1.out')
#biasData = np.loadtxt('Q4_P2_bias1.out')
#weightData2nd = np.loadtxt('Q4_P2_weights2.out')
#biasData2nd = np.loadtxt('Q4_P2_bias2.out')
#weightData = np.loadtxt('Q4_P3_weights1.out')
#biasData = np.loadtxt('Q4_P3_bias1.out')
#weightData2nd = np.loadtxt('Q4_P3_weights2.out')
#biasData2nd = np.loadtxt('Q4_P3_bias2.out')
env = gym.make('SpaceInvadersNoFrameskip-v0')
env = wrappers.Monitor(env, '/10703-assignment2-YHuang-YZhou', force = True)
env.render()

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
        actionMatrix = np.add(np.add(np.matmul(obsMatrix, weightData), biasData), np.add(np.matmul(obsMatrix, weightData2nd), biasData2nd)) #linearQ4 action selector
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