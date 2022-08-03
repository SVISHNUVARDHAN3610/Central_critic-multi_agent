import torch
import torch.optim as optim
import numpy as np
import pickle as p
import csv
import matplotlib.pyplot as plt
from Network.network import Central
from Agents.Agent1 import Agent1
from Agents.Agent2 import Agent2
from make_env import make_env
env = make_env("simple_reference")

class Mappo:
  def __init__(self,state_size,action_size,reward_size,n_agents,n_games , step,buffer):
    self.state_size,self.action_size,self.reward_size = state_size,action_size,reward_size
    self.n_agents,self.n_games,self.step              = n_agents,n_games,step
    self.device      = torch.device("cpu")
    self.buffer      = buffer
    self.agent1      = Agent1(self.state_size,self.action_size,self.reward_size,self.buffer)
    self.agent2      = Agent2(self.state_size,self.action_size,self.reward_size,self.buffer)
    self.critic      = Central(self.state_size,self.action_size,self.reward_size).to(self.device)
    self.critic_optim= optim.Adam(self.critic.parameters() ,lr = 0.00000007)
    self.path        = ["memory/central_critic.pth"]
  def choose_actions(self,state,i):
    actions   = []
    obs1      = torch.from_numpy(state[0]).float().to(self.device)
    obs2      = torch.from_numpy(state[1]).float().to(self.device)
    act1      = self.agent1.choose_action(obs1,i)
    act2      = self.agent2.choose_action(obs2,i)
    actions.append(act1)
    actions.append(act2)
    return actions
  def q_value(self,state,action,reward,i):
    state      = torch.from_numpy(state).float().to(self.device)
    action     = torch.tensor(action,dtype = torch.float32).to(self.device)
    reward     = torch.tensor([reward,0],dtype = torch.float32).to(self.device)
    q_value    = self.critic(state,action,reward,i).to(self.device)
    return q_value
  def appending(self,value,returns,g_state,g_next_state,loss):
    self.buffer.cc_loss.append(loss)
    self.buffer.cc_state.append(g_state)
    self.buffer.cc_next_state.append(g_next_state)
    self.buffer.cc_value.append(value)
    self.buffer.cc_returns.append(returns)
    self.buffer.agent1_returns = []
    self.buffer.agent2_returns = []
  def update(self,state,next_state,reward,done):
    g_state    = state[0] + state[1]
    g_next     = next_state[0] + next_state[1]
    g_reward   = reward[0] + reward[1]
    actions    = self.choose_actions(state,0)
    n_actions  = self.choose_actions(next_state,1)
    action     = actions[0]+[1]
    next_action= self.choose_actions(next_state,1)
    c_value    = self.q_value(g_state,action,g_reward,0)
    a1_nvalue  = self.q_value(next_state[0],n_actions[0],reward[0],1)
    a2_nvalue  = self.q_value(next_state[1],n_actions[1],reward[1],1)
    self.agent1.learn(state[0],next_state[0],reward[0],a1_nvalue,done[0])
    self.agent2.learn(state[1],next_state[1],reward[1],a2_nvalue,done[0])
    returnsa1  = self.buffer.agent1_returns[0].to(self.device)
    returnsa2  = self.buffer.agent2_returns[0].to(self.device)
    returns    = returnsa1/2+returnsa2/2
    loss       = (returns - c_value).mean()**2
    self.appending(c_value,returns,g_state,g_next,loss)
    torch.save(self.critic,self.path[0])
    self.critic_optim.zero_grad()
    loss.backward()
    self.critic_optim.step()
  def mean(self,i):
      self.buffer.agent1_mean.append(sum(self.buffer.agent1_reward)/len(self.buffer.agent1_reward))
      self.buffer.agent2_mean.append(sum(self.buffer.agent2_reward)/len(self.buffer.agent2_reward))
      self.buffer.agent1_mean_loss.append(sum(self.buffer.agent1_loss)/len(self.buffer.agent1_loss))
      self.buffer.agent2_mean_loss.append(sum(self.buffer.agent2_loss)/len(self.buffer.agent2_loss))
      self.buffer.agent1_mean_returns.append(sum(self.buffer.agent1_main_returns)/len(self.buffer.agent1_main_returns))
      self.buffer.agent2_mean_returns.append(sum(self.buffer.agent2_main_returns)/len(self.buffer.agent2_main_returns))
      self.buffer.agent1_mean_loss.append(sum(self.buffer.agent1_loss)/len(self.buffer.agent1_loss))
      self.buffer.agent2_mean_loss.append(sum(self.buffer.agent2_loss)/len(self.buffer.agent2_loss))
      self.buffer.cc_mean_loss.append(sum(self.buffer.cc_loss)/len(self.buffer.cc_loss))
      self.buffer.agent1_mean_returns.append(sum(self.buffer.agent1_main_returns)/len(self.buffer.agent1_main_returns))
      self.buffer.agent2_mean_returns.append(sum(self.buffer.agent2_main_returns)/len(self.buffer.agent2_main_returns))
      self.buffer.agent1_log_prob_mean.append(sum(self.buffer.agent1_log_prob)/len(self.buffer.agent1_log_prob))
      self.buffer.agent2_log_prob_mean.append(sum(self.buffer.agent2_log_prob)/len(self.buffer.agent2_log_prob))
      self.buffer.agent1_next_log_probm.append(sum(self.buffer.agent1_next_log_prob)/len(self.buffer.agent1_next_log_prob))
      self.buffer.agent2_next_log_probm.append(sum(self.buffer.agent2_next_log_prob)/len(self.buffer.agent2_next_log_prob))
      self.buffer.cc_mean_loss.append(sum(self.buffer.cc_loss)/len(self.buffer.cc_loss))
      self.buffer.cc_mean_value.append(sum(self.buffer.cc_value)/len(self.buffer.cc_value))
      self.buffer.episodes.append(i)
  def saving(self,m):
    l = []
    for i in range(len(self.buffer.agent1_mean)):
      l.append(i)
    st_line = ["slno","agent1_reward","agent2_reward","agent1_loss","agent2_loss","agent1_value","agent2_value","agent1_return","agent2_returns","agent1_log_prob","agent2_log_prob","agent1_next_log_prob","agent2_next_log_prob","centeral_value","central_loss"]
    with open("memory/multiagent.csv","w") as mcsv:
      write = csv.writer(mcsv)
      write.writerow([st_line[0],st_line[1],st_line[2],st_line[3],st_line[4],st_line[5],st_line[6],st_line[7],st_line[8],st_line[9],st_line[10],st_line[11],st_line[12],st_line[13],st_line[14]])
      write.writerow([l[m],self.buffer.agent1_mean[m],self.buffer.agent2_mean[m],self.buffer.agent1_mean_loss[m],self.buffer.agent2_mean_loss[m],self.buffer.agent1_value_mean[m],self.buffer.agent2_value_mean[m],self.buffer.agent1_mean_returns[m],self.buffer.agent2_mean_returns[m],self.buffer.agent1_log_prob_mean[m],self.buffer.agent2_log_prob_mean[m],self.buffer.agent1_next_log_probm[m],self.buffer.agent2_next_log_probm[m],self.buffer.cc_mean_value[m],self.buffer.cc_mean_loss[m]])
  def pickeling(self):
    p.dump(self.buffer.agent1_mean,open("memory/agent1.bat","wb"))
    p.dump(self.buffer.agent2_mean,open("memory/agent2.bat","wb"))
    p.dump(self.buffer.agent1_mean_loss,open("memory/agent1_loss.bat","wb"))
    p.dump(self.buffer.agent2_mean_loss,open("memory/agent2_loss.bat","wb"))
    p.dump(self.buffer.agent1_value_mean,open("memory/agent1_value.bat","wb"))
    p.dump(self.buffer.agent2_value_mean,open("memory/agent2_value.bat","wb"))
    p.dump(self.buffer.agent1_mean_returns,open("memory/agent1_value.bat","wb"))
    p.dump(self.buffer.agent2_mean_returns,open("memory/agent2_value.bat","wb"))
    p.dump(self.buffer.agent1_log_prob_mean,open("memory/agent1_log_prob.bat","wb"))
    p.dump(self.buffer.agent2_log_prob_mean,open("memory/agent2_log_prob.bat","wb"))
    p.dump(self.buffer.agent1_next_log_probm,open("memory/agent1_next_log_prob.bat","wb"))
    p.dump(self.buffer.agent2_next_log_probm,open("memory/agent2_next_log_prob.bat","wb"))
    p.dump(self.buffer.agent1_next_value_m,open("memory/agent1_next_value.bat","wb"))
    p.dump(self.buffer.agent2_next_value_m,open("memory/agent2_next_value.bat","wb"))
    p.dump(self.buffer.cc_loss,open("memory/cc_loss.bat","wb"))
    p.dump(self.buffer.cc_value,open("memory/cc_value.bat","wb"))
  def ploting(self,i):
    plt.plot(self.buffer.episodes,self.buffer.agent1_mean)
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.title("episodes vs rewards")
    if i%2==0:
      plt.savefig("memory/episodes_vs_rewards.png")
  def run(self):
    for i in range(self.n_games):
      state = env.reset()
      score = [0*2]
      done  = [False*2]
      for step in range(self.step):
        action = self.choose_actions(state,0)
        next_state,reward,done,_ = env.step(action)
        env.render()
        if done:    
          self.update(state,next_state,reward,done)          
        else:
          self.update(state,next_state,reward,done)
      self.mean(i)
      print("episode:",i,"agent1_reward:",self.buffer.agent1_mean[i],"agent2_reward",self.buffer.agent2_mean[i])
      self.pickeling()
      self.ploting(i)
      #self.saving(i)
  