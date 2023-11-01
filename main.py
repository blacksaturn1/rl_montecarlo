import random
import numpy as np
from helper import Helper
import matplotlib.pyplot as plt

states=[0, 1, 2, 3, 4, 5]

actions=[0,1]
#actionValues=[-1,0,1]
actionValues=[-1,1]

probability=[0.05, .15, .8]
endStates=[0, 5]
rewards = [1, 0, 0, 0, 0, 5]

l=.95
epsilon=.95
value_function = np.zeros( len(states) )
returns = [dict() for x in range (len(states))]
Q=np.zeros( (len(states),len(actions)) )
policy = .5* np.ones( (len(states),len(actions)) )
episodes=[]
mdp=Helper()
maxEpisodes=1000
debug=False
value_functions=[]

def reset():
    value_function = np.zeros( len(states) )
    returns = [dict() for x in range (len(states))]
    Q=np.zeros( (len(states),len(actions)) )
    policy = np.zeros( (len(states),len(actions)) )
    episodes=[]
    
def getRandomEpisode():
    end=False
    episode=[]
    startState=random.randrange(1,5,1)
    state=startState
    count = 0
    startAction=random.randrange(0,2,1)
    while not end:
        count+=1
        # end episode if state is 0 or 5
        if (state==0 or state==5):
            end = True
            c= (state,999,0)
            if debug is True:
                print(c); print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            break

        if count >1:
            # break ties randomly
            action=np.random.choice(np.flatnonzero(policy[state,:] == policy[state,:].max()))
            #action=np.argmax(policy[state,:])
        else:
            action=startAction
        u=actionValues[action]
        actionProbability=random.random()
        if(actionProbability<=probability[0]):
            u=u*-1
        elif(actionProbability<probability[1]):
            u=0
        else:
            u=u

        nextState = state+u
        reward = rewards[nextState]
        c= (state,action,reward)
        if debug is True:
            print(c); 
            #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        episode.append(c)
        state=nextState
        
    return episode

def getEpisodeStartInMiddleOnPolicy():
    end=0
    episode=[]
    startState=4
    state=startState
    while end == 0 :
        action=np.argmax(policy[state,:])
        u=actionValues[action]
        actionProbability=random.random()
        if(actionProbability<=probability[0]):
            u=u*-1
        elif(actionProbability<probability[1]):
            u=0
        else:
            u=u
        nextState = state+u
        reward = rewards[nextState]
        c= (state,action,reward)
        if debug is True:
            print(c); print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        episode.append(c)
        state=nextState
        if (state==0 or state==5):
            end = 1
    return episode


def inList(episode,target_s,target_a):
    for state,action,_ in episode:
        if state == target_s and action == target_a:
            return True
    return False

def state_in_List(episode,target_s):
    for state,_,_ in episode:
        if state == target_s :
            return True
    return False


def MonteCarloExploringStart(Q,episode):
    G=0
    e=episode.copy()
    while len(e)>0:
        t=e.pop()
        if( len(t)>1):
            (state,action,reward)=t
            G = l*G + reward
            in_list = inList(e,state,action)
            
            if not in_list:
                if action not in returns[state]:
                    returns[state][action]=[]    
                returns[state][action].append(G)

                Q[state,action]=sum(returns[state][action])/len(returns[state][action])
                maxValueIndex = np.argmax(Q[state,:])
                statesActionValues=Q[state,:]
                indicesOfNewMax = np.where(statesActionValues==statesActionValues[maxValueIndex])[statesActionValues.ndim-1]
                # y=np.sort(indicesOfNewMax)
                policy[state,:]=np.zeros((len(actions)))
                for i in indicesOfNewMax:
                    policy[state,i] = 1 / len(indicesOfNewMax)
    
    value_function = mdp.getStateValue(Q)
    return Q,value_function
    #print(policy)
    

def MonteCarloStateValuePrediction(V,episode):
    G=0
    #e = getRandomEpisode()
    e=episode.copy()

    while len(e)>0:
        t=e.pop()
        if( len(t)>1):
            (state,action,reward)=t
            G = l*G + reward
            in_list = state_in_List(e,state)
            
            if in_list is False:
                #V[state]=
                returns[state][action].append(G)

                Q[state,action]=sum(returns[state][action])/len(returns[state][action])
                maxValueIndex = np.argmax(Q[state,:])
                statesActionValues=Q[state,:]
                indicesOfNewMax = np.where(statesActionValues==statesActionValues[maxValueIndex])[statesActionValues.ndim-1]
                y=np.sort(indicesOfNewMax)
                policy[state,:]=np.zeros((len(actions)))
                for i in indicesOfNewMax:
                    policy[state,i] = 1 / len(indicesOfNewMax)
    
    value_function = mdp.getStateValue(Q)
    return Q,value_function
    

def MonteCarloOnPolicy(Q):
    G=0
    e = getEpisodeStartInMiddleOnPolicy()
    while len(e)>0:
        t=e.pop()
        if( len(t)>1):
            (state,action,reward)=t
            G = l*G + reward
            in_list = inList(e,state,action)
            
            if not in_list:
                if action not in returns[state]:
                    returns[state][action]=[]    
                returns[state][action].append(G)

                Q[state,action]=sum(returns[state][action])/len(returns[state][action])
                maxValueIndex = np.argmax(Q[state,:])
                statesActionValues=Q[state,:]
                indicesOfNewMax = np.where(statesActionValues==statesActionValues[maxValueIndex])[statesActionValues.ndim-1]
                y=np.sort(indicesOfNewMax)
            
                policy[state,:]=np.zeros((len(actions)))

                for i in indicesOfNewMax:
                    policy[state,i] = 1 / len(indicesOfNewMax)
    
    value_function = mdp.getStateValue(Q)
    return Q,value_function

for count,episode in enumerate(range(maxEpisodes)):
    episode=getRandomEpisode()
    Q, value_function = MonteCarloExploringStart(Q,episode)
    value_functions.append(value_function)
    if (count+1) % 10000 == 0: print("Completed episode: {}".format(count+1))
    


yPlots=[list() for x in states]
for value_function_episode in value_functions:
    for index,value in enumerate(value_function_episode):
        yPlots[index].append(value)

x = np.arange(1,len(value_functions)+1)
for index,values in enumerate(yPlots):
    plt.plot(x,values,label = "State {}".format(index))
plt.legend()
plt.show()

input("Press enter to end")

# reset
reset()
exit()
# Start with arbitrary e-soft policy, hence 50% left or right
policy = np.ones( (len(states),len(actions)) )*.5

for count,episode in enumerate(range(maxEpisodes)):
    Q, value_function = MonteCarloOnPolicy(Q)
    value_functions.append(value_function)
    if (count+1) % 10000 == 0: print("Completed episode: {}".format(count+1))




yPlots=[list() for x in states]
for value_function_episode in value_functions:
    for index,value in enumerate(value_function_episode):
        yPlots[index].append(value)

x = np.arange(1,len(value_functions)+1)
for index,values in enumerate(yPlots):
    plt.plot(x,values,label = "State {}".format(index))
plt.legend()

plt.show(block=True)
