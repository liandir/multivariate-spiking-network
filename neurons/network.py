import numpy as np
import torch
# import threading
from collections import OrderedDict

from .activations import *
from .bundles import *



class History:
    
    def __init__(self):
        self._current = []
        self._voltage = []
        self._outputs = []
    
    @property
    def current(self):
        return np.stack(self._current)
    
    @property
    def voltage(self):
        return np.stack(self._voltage)
    
    @property
    def outputs(self):
        return np.stack(self._outputs)
    
    def reset(self):
        self._current = []
        self._voltage = []
        self._outputs = []
    
    
class Network:
    
    def __init__(self, input_size=1):
        self.input_size = input_size
        
        self.bundles = OrderedDict()
        self.connections = OrderedDict()
        
        self.compiled = False
        self._train = False
        self._history = True
        
        self.device = "cpu"
        
    def add(self, ntype, name, size, **kwargs):
        if name not in self.bundles:
            self.bundles[name] = ntype(size, **kwargs)
        else:
            print(f"bundle with name '{name}' already exists.")
            
    def remove(self, name):
        if name in self.bundles:
            self.bundles.pop(name)
        else:
            print(f"bundle with name '{name}' does not exist.")
        
    def connect(self, bundle_in, bundle_out):
        if bundle_in in self.bundles and bundle_out in self.bundles:
            if bundle_out in self.connections:
                self.connections[bundle_out] += [bundle_in]
            else:
                self.connections[bundle_out] = [bundle_in]
        else:
            if bundle_in not in self.bundles and bundle_out not in self.bundles:
                print(f"'{bundle_in}' and '{bundle_out}' do not exist.")
            elif bundle_in not in self.bundles:
                print(f"'{bundle_in}' does not exist.")
            elif bundle_out not in self.bundles:
                print(f"'{bundle_out}' does not exist.")
            
    def disconnect(self, bundle_in, bundle_out):
        if bundle_in in self.bundles and bundle_out in self.bundles:
            self.connections[bundle_out] -= {bundle_in}
        else:
            if bundle_in not in self.bundles and bundle_out not in self.bundles:
                print(f"'{bundle_in}' and '{bundle_out}' do not exist.")
            elif bundle_in not in self.bundles:
                print(f"'{bundle_in}' does not exist.")
            elif bundle_out not in self.bundles:
                print(f"'{bundle_out}' does not exist.")
    
    @property
    def train(self):
        return self._train
    
    @train.setter
    def train(self, active):
        for bundle in self.bundles:
            self.bundles[bundle].train = active
        self._train = active
    
    @property
    def record_history(self):
        return self._history
    
    @record_history.setter
    def record_history(self, active):
        self._history = active
            
    def reset_history(self):
        for name in self.bundles:
            self.history[name].reset()
    
    def compile(self, device="cpu"):
        '''initialize synapses and inputs. also check for coherence of model architecture.'''
        
        if not self.compiled:
            for head in self.connections:
                n_inputs = np.sum([self.bundles[tail].size for tail in self.connections[head]])
                self.bundles[head].inputs = torch.zeros(
                    n_inputs,
                    4
                )
                self.bundles[head].synapses = torch.tensor(
                    np.random.uniform(
                        0,
                        1 / n_inputs,
                        size=(self.bundles[head].size, n_inputs, 4)
                    ).astype(np.float32)
                )
                self.bundles[head].compiled = True
            
            self.compiled = True
            for name in self.bundles:
                if self.bundles[name].ntype != 'Sensory':
                    if self.bundles[name].inputs is None:
                        self.compiled = False

            if self.compiled:
                self.history = {}
                for name in self.bundles:
                    self.history[name] = History()
                
                self.to(device)
                            
            if self.compiled:
                print('model successfully compiled.\n')
                # print(self)
            else:
                print('model was not compiled. some bundles are not connected.')
        
        else:
            print('model has already been compiled.')
    
    def to(self, device):
        self.device = device
        for name in self.bundles:
            self.bundles[name].device = device
            if self.bundles[name].ntype != 'Sensory':
                self.bundles[name].transmitter = self.bundles[name].transmitter.to(device)
                self.bundles[name].transmitter_mask = self.bundles[name].transmitter_mask.to(device)
                self.bundles[name].inputs = self.bundles[name].inputs.to(device)
                self.bundles[name].synapses = self.bundles[name].synapses.to(device)
                self.bundles[name].current = self.bundles[name].current.to(device)
                self.bundles[name].voltage = self.bundles[name].voltage.to(device)
                self.bundles[name].outputs = self.bundles[name].outputs.to(device)
            else:
                self.bundles[name].transmitter = self.bundles[name].transmitter.to(device)
                self.bundles[name].stim = self.bundles[name].stim.to(device)
    
    def step(self, dt=0.001):
        '''single time step. requires model to be compiled.'''
        # update inputs of bundles
        for head in self.connections:
            self.bundles[head].inputs[:] = torch.cat([self.bundles[tail].outputs for tail in self.connections[head]])
        
        # update neuron states
        # threads = []
        for name in self.bundles:
            if self.bundles[name].ntype != 'Sensory':
                # thread = threading.Thread(target=self.bundles[name].step, args=[dt])
                # thread.start()
                # threads.append(thread)
                self.bundles[name].step(dt=dt)
                if self.record_history:
                    self.history[name]._current += [self.bundles[name].current.clone().cpu()]
                    self.history[name]._voltage += [self.bundles[name].voltage.clone().cpu()]
                    self.history[name]._outputs += [self.bundles[name].outputs.clone().cpu()]
        
        # for thread in threads:
        #     thread.join()
    
    def __repr__(self):
        out = f"Network Summary\n"
        out += '='*64 + '\n'
        out += 'Bundles:\n'
        out += '_'*64 + '\n'
        out += 'name\t\tntype\t\tunits\n'
        out += '-'*64 + '\n'
        for name in self.bundles:
            out += f'{name}\t\t{self.bundles[name].ntype}\t\t{self.bundles[name].size}\n'
        if len(self.connections) > 0:
            out += '='*64 + '\n'
            out += 'Connections:\n'
            out += '_'*64 + '\n'
            for name in self.connections:
                for inc in self.connections[name]:
                    out += f'{inc} -> {name}\n'
        out += '='*64 + '\n'
        out += f'model compiled: {self.compiled}'
        return out
    
    
