import torch, numpy, gym, time, random
import torch.nn as nn
import torch.optim as opt
from torch.distributions import Categorical
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]
network = nn.Sequential(nn.Linear(observation_space,24),nn.ReLU(),nn.Linear(24,action_space),nn.Softmax(dim=-1))
criterion = nn.MSELoss()
optimizer = opt.SGD(network.parameters(),lr=0.00025)

epochs = 60000
max_epsilon = 1.0
min_epsilon = 0.0
percent = 0.2
decay_rate = 1/(epochs*percent)
epsilon = max_epsilon
discount_factor = 0.9
log_actions = torch.tensor([])
episode_rewards = []
eps =numpy.finfo(float).eps

def preprocess(state):
    return torch.tensor(state).float()

def next_action(state):
    global log_actions

    distribution = network(state)
    #c = Categorical(distribution)
    #action = c.sample()
    #log = torch.tensor([0.0]).add(c.log_prob(action))

    if random.random()>epsilon:
        val,action = distribution.max(0)
        action = action.item()
        log = torch.tensor([0.0]).add(torch.log(val))
    else:
        action = random.randint(0,action_space-1)
        log = torch.tensor([0.0]).add(torch.log(torch.tensor(distribution[action])))

    if len(log_actions.data)==0:
        log_actions = log
    else:
        log_actions = torch.cat([log_actions,log])
    return action

def select_action(state):
    out = network(state)
    return out.max(0)

def discount_rewards(rewards):
    discounted_rewards = []
    run = 0

    for i in reversed(rewards):
        run = i + run*discount_factor
        discounted_rewards.insert(0, run)
    return discounted_rewards

def update_policy():
    global episode_rewards, log_actions

    discounted_rewards = discount_rewards(episode_rewards)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + eps)
    loss = torch.sum(discounted_rewards*log_actions*-1,-1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_rewards=[]
    log_actions = torch.tensor([])
    return loss

def decay():
    return max(min_epsilon,epsilon-decay_rate)

print("---------TRAINING---------")
mr = 0
rewardmeans = []
lossmeans = []
loss = 0
for e in range(epochs):
        frames = 0
        runreward = 0
        state = env.reset()
        state = preprocess(state)
        l = 0
        while True:
            frames += 1
            action = next_action(state)
            new_state, reward, done, i = env.step(action)
            new_state = preprocess(new_state)

            runreward+=reward
            state = new_state
            episode_rewards.append(reward)
            if done or frames%200 == 0:
                #print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                mr += runreward
                loss += update_policy()
                break
        epsilon = decay()
        if (e+1)%100==0:
            mean = mr/100
            rewardmeans.append(float(mean))
            mr = 0
            lossmeans.append(float(loss / 100))
            loss = 0
            print(str(e + 1) + " M: " + str(mean)+" E: "+str(epsilon))
            if mean>=-110.0:
                break

print("---------TESTING---------")
epsilon=0
for e in range(5):
        state = env.reset()
        runreward = 0
        state = preprocess(state)
        frames=0
        while True:
            frames+=1
            prob ,action = select_action(state)
            new_state, reward, done, i = env.step(action.item())
            new_state = preprocess(new_state)
            env.render()
            time.sleep(0.02)
            runreward += reward
            state = new_state
            if done or frames%200 == 0:
                print("E: "+str(e)+" F: "+str(frames)+" R: "+str(runreward)+" Eps: "+str(epsilon))
                break

print("---------PLOTTING---------")
plt.figure(0)
plt.plot(lossmeans)
plt.title("Mean Loss")
#plt.savefig('./Plots/PolicyGradients/meanloss.png',bbox_inches='tight')
plt.figure(1)
plt.plot(rewardmeans,color="orange")
plt.title("Mean Reward")
#plt.savefig('./Plots/PolicyGradients/meanreward.png',bbox_inches='tight')
plt.show()