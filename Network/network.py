import torch
import torch.nn as nn
import torch.nn.functional as f

class Actor(nn.Module):
  def __init__(self,state_size,action_size):
    super(Actor,self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.fc2 = nn.Linear(32,64)
    self.fc3 = nn.Linear(64,128)
    self.fc4 = nn.Linear(128,512)
    self.fc5 = nn.Linear(512,128)
    self.fc6 = nn.Linear(128,self.action_size)
  def forward(self,x,i):
    if i ==0:
      L = self.state_size
    else:
      L = 11
    fc1 = nn.Linear(L,32)
    a1 = fc1(x)
    a2 = f.relu(self.fc5(f.relu(self.fc4(f.relu(self.fc3(f.relu(self.fc2(f.relu(a1)))))))))
    a3 = f.softmax(self.fc6(a2))
    return a3
class Critic(nn.Module):
  def __init__(self,state_size,action_size,reward_size):
    super(Critic,self).__init__()
    self.state_size,self.action_size ,self.reward_size = state_size,action_size,reward_size
    self.fac    = nn.Linear(self.action_size,32)
    self.frc    = nn.Linear(self.reward_size,32)
    self.fc3    = nn.Linear(64,128)
    self.fc4    = nn.Linear(128,512)
    self.fc5    = nn.Linear(512,128)
    self.fc6    = nn.Linear(128,1)
  def forward(self,s,a,r,i):
    if i ==0:
      L = self.state_size
    else:
      L = 11
    fc1    = nn.Linear(L,32)
    a1     = fc1(s)
    a2     = self.fac(a)
    a3     = self.frc(r)
    x      = torch.cat([a1.view(32,-1),a2.view(32,-1),a3.view(32,-1)])
    x      = torch.reshape(x,(-1,))
    fc2    = nn.Linear(x.shape[0],64)
    f1     = f.relu(self.fc5(f.relu(self.fc4(f.relu(self.fc3(f.relu(fc2(f.relu(x)))))))))
    f2     = self.fc6(f1)
    return f2
class Central(nn.Module):
  def __init__(self,state_size,action_size,reward_size):
    super(Central,self).__init__()
    self.state_size,self.action_size ,self.reward_size = state_size,action_size,reward_size
    self.fac    = nn.Linear(self.action_size,32)
    self.frc    = nn.Linear(self.reward_size,32)
    self.fc3    = nn.Linear(64,128)
    self.fc4    = nn.Linear(128,512)
    self.fc5    = nn.Linear(512,128)
    self.fc6    = nn.Linear(128,1)
  def forward(self,s,a,r,i):
    if i ==0:
      L = self.state_size
    else:
      L = 11
    fc1    = nn.Linear(L,32)
    a1     = fc1(s)
    a2     = self.fac(a)
    a3     = self.frc(r)
    x      = torch.cat([a1.view(32,-1),a2.view(32,-1),a3.view(32,-1)])
    x      = torch.reshape(x,(-1,))
    fc2    = nn.Linear(x.shape[0],64)
    f1     = f.relu(self.fc5(f.relu(self.fc4(f.relu(self.fc3(f.relu(fc2(f.relu(x)))))))))
    f2     = self.fc6(f1)
    return f2