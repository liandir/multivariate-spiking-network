import torch
from .activations import *


class LIFrate:
    
    def __init__(self,
        size,             # number of neurons
        v_rest=0.0,       # membrane resting potential
        v_thres=0.0,      # membrane threshold potential
        tau_v=0.1,        # membrane potential time constant
        tau_s_ex=100.0,     # synapse time constant
        tau_s_in=100.0,     # 
        tau_s_ex_f=1000.0,     # synapse time constant
        tau_s_in_f=1000.0,
        R=280.0,            # membrane resistance
        activation=f5,
        gamma=1.0,
        delta=0.1,
        theta=0.001,
        transmitter=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    ):  
        self.size = size
        self.ntype = 'LIFrate'
        
        # parameters
        self.v_rest = v_rest
        self.v_thres = v_thres
        self.tau_v = tau_v
        self.tau_s_ex = tau_s_ex
        self.tau_s_in = tau_s_in
        self.tau_s_ex_f = tau_s_ex_f
        self.tau_s_in_f = tau_s_in_f
        self.R = R
        self.activation = activation
        self.gamma = gamma
        self.delta = delta
        self.theta = theta
        
        self.transmitter = transmitter
        self.transmitter_mask = torch.tensor([1.0, -1.0, 1.0, -1.0])
        
        # variables
        self.current = torch.zeros(self.size)
        self.voltage = self.v_rest * torch.ones(self.size)
        self.outputs = torch.zeros(self.size, 4)
        
        self.inputs = None
        self.synapses = None
        
        self.compiled = False
        self.train = False
        self.device = "cpu"
        
    def step(self, dt=0.001):
        currents = torch.einsum('ijk,jk->ik', self.synapses, self.inputs)
        
        if self.train:
            dr = self.gamma * self.activation(self.gamma * (self.voltage - self.v_thres), deriv=True, device=self.device)
            
            ex_feedback = self.activation(self.delta * self.R * currents[:,2], device=self.device)
            in_feedback = self.activation(self.delta * self.R * currents[:,3], device=self.device)
            
            self.synapses[:,:,0] += ((torch.outer(self.inputs[:,0], ex_feedback) - self.synapses[:,:,0].T * in_feedback) * dr).T / self.tau_s_ex * dt
            self.synapses[:,:,1] += ((torch.outer(self.inputs[:,1], in_feedback) - self.synapses[:,:,1].T * ex_feedback) * dr).T / self.tau_s_in * dt
            
#             ex_current = self.activation(self.delta * self.R * (currents[:,0] - currents[:,1]))
#             in_current = self.activation(self.delta * self.R * (currents[:,1] - currents[:,0]))
            
#             self.synapses[:,:,2] -= (self.synapses[:,:,2].T * ex_current).T / self.tau_s_ex_f * dt
#             self.synapses[:,:,3] -= (self.synapses[:,:,3].T * in_current).T / self.tau_s_in_f * dt
        
        self.current[:] = torch.sum(currents * self.transmitter_mask, dim=1)
        self.voltage += (self.v_rest - self.voltage + self.R * self.current) / self.tau_v * dt
        self.outputs[:] = torch.outer(self.activation(self.gamma * (self.voltage - self.v_thres), device=self.device), self.transmitter)
    
    def __repr__(self):
        if self.compiled:
            return f'LIFrateBundle(size={self.size}, inputs={self.inputs.shape})'
        else:
            return f'LIFrateBundle(size={self.size}, inputs={self.inputs})'

    
class Sensory:
    
    def __init__(self, size, transmitter=torch.tensor([1.0, 0.0, 0.0, 0.0])):
        self.size = size
        self.ntype = 'Sensory'
        self.transmitter = transmitter
        self.stim = torch.zeros(self.size)
        self.device = "cpu"
    
    @property
    def outputs(self):
        return torch.outer(self.stim, self.transmitter)
        
    def __repr__(self):
        return f'SensoryBundle(size={self.size})'
    