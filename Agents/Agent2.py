import torch
import torch.optim as optim
import numpy as np
from Network.network import Actor,Critic

class Agent2:
  def __init__(self,state_size,action_size,reward_size,buffer):
    self.state_size = state_size
    self.action_size = action_size
    self.reward_size = reward_size
    self.gamma       = 0.99
    self.lamda       = 0.95
    self.lr1         = 0.00000000001
    self.lr2         = 0.00000000007
    self.buffer      = buffer
    self.device      = torch.device("cpu")
    self.actor       = Actor(self.state_size ,self.action_size).to(self.device)
    self.critic      = Critic(self.state_size,self.action_size,self.reward_size).to(self.device)
    self.actor_optim = optim.Adam(self.actor.parameters() , lr = self.lr1)
    self.critic_optim= optim.Adam(self.critic.parameters() ,lr = self.lr2)
    self.path        = ["memory/Agent2_Actor.pth","memory/Agent2_Critic.pth"]

  def choose_action(self,state,i):
    ar = np.random.uniform(0.271,0.519,1)[0]
    br = np.random.uniform(0.289,0.515,1)[0]
    cr = np.random.uniform(0.289,0.515,1)[0]
    dr = np.random.uniform(0.281,0.519,1)[0]
    fr = np.random.uniform(0.281,0.519,1)[0]
    rand              = torch.tensor([ar,br,cr,dr,fr],dtype = torch.float32)
    act               = self.actor(state,i).to(self.device)
    action            = act + rand
    return action.detach().numpy()
  def q_value(self,state,action,reward,i):
    reward            = torch.tensor([reward , 0],dtype = torch.float32).to(self.device)
    action            = torch.tensor(action,dtype = torch.float32).to(self.device)
    q_value           = self.critic(state,action,reward,i).to(self.device)
    return q_value
  def get_gae(self,reward,d,value,next_value):
    returns  = []
    gae      = 0
    reward   = torch.tensor(reward,dtype = torch.float32).to(self.device)
    for i in range(5):
      delta  = reward + (1-d)*self.gamma*next_value -value
      gae    = delta + self.lamda * self.gamma*(1-d)*gae
      returns.insert(0,gae + value + next_value)
    return returns
  def appending(self,state,next_state,reward,value,log_prob,next_log_prob,loss,returns):
    self.buffer.agent2_state.append(state)
    self.buffer.agent2_next_state.append(next_state)
    self.buffer.agent2_reward.append(reward)
    self.buffer.agent2_value.append(value)
    self.buffer.agent2_log_prob.append(log_prob)
    self.buffer.agent2_next_log_prob.append(next_log_prob)
    self.buffer.agent2_loss.append(loss)
    self.buffer.agent2_returns.append(returns)
    self.buffer.agent2_main_returns.append(returns)
  def learn(self,state,next_state,reward,next_value,done):
    state             = torch.from_numpy(state).float().to(self.device)
    next_state        = torch.from_numpy(next_state).float().to(self.device)
    action            = self.choose_action(state,0)
    next_action       = self.choose_action(next_state,1)
    log_prob          = torch.log(torch.tensor(action,dtype = torch.float32)).to(self.device)
    next_log_prob     = torch.log(torch.tensor(next_action,dtype = torch.float32)).to(self.device)
    value             = self.q_value(state,action,reward,0)
    next_value        = next_value
    ratio             = torch.exp(next_log_prob - log_prob)
    r                 = self.get_gae(reward,done,value,next_value)
    returns           = torch.tensor([r[0][0],r[1][0],r[2][0],r[3][0],r[4][0]])
    advantage         = returns + value + next_value
    s1                = ratio * advantage
    s2                = torch.clamp(ratio , 0.8,1.2)*advantage
    actor_loss        = torch.min(s1,s2).mean()
    critic_loss       = (returns - value).mean()**2
    loss              = actor_loss + critic_loss*0.5
    self.appending(state,next_state,reward,value,loss,log_prob,next_log_prob,returns)
    torch.save(self.actor,self.path[0])
    torch.save(self.critic ,self.path[1])
    self.actor_optim.zero_grad()
    self.critic_optim.zero_grad()
    loss.backward()
    self.actor_optim.step()
    self.critic_optim.step()