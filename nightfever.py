import csv
import random
import time
import numpy as np
import torch
import torch.nn as nn
import songwriter


## Global variables

P_EPS = 1e-6



## Classes

class NightFever(nn.Module):
	""" Core model with the three RNN circuit """

	def __init__(self, n_dim, dev, sizes=None, batch_size=None):

		super(NightFever, self).__init__()

		if sizes is None:
			n_hidden_perception = 2 * n_dim   # P smallest
			n_hidden_slow = 4 * n_dim         # H small, limited capacity
			n_hidden_fast = 32 * n_dim        # C much larger
		else:
			n_hidden_perception, n_hidden_slow, n_hidden_fast = sizes

		self.n_hidden_perception = n_hidden_perception
		self.n_hidden_slow = n_hidden_slow
		self.n_hidden_fast = n_hidden_fast
		self.n_dim = n_dim
		self.dev   = dev
		self.batch_size = batch_size

		self.perception = RNN(n_dim, 1, n_hidden_perception, dev, batch_size)
		self.fast = RNN(n_dim, 1, n_hidden_fast, dev, batch_size)
		self.slow = RNN(n_dim, 2, n_hidden_slow, dev, batch_size)

		# Each RNN takes and returns an n_dim vector, so every link is n_dim -> n_dim.
		self.perc_fast = nn.Linear(n_hidden_perception, n_hidden_fast)   # perception -> fast
		self.perc_slow = nn.Linear(n_hidden_perception, n_hidden_slow)   # perception -> slow
		self.fast_slow = nn.Linear(n_hidden_fast, n_hidden_slow)   # fast -> slow
		self.fast_perc = nn.Linear(n_hidden_fast, n_hidden_perception)   # fast -> perception
		self.slow_perc = nn.Linear(n_hidden_slow, n_hidden_perception)   # slow -> perception
		self.slow_fast = nn.Linear(n_hidden_fast, n_hidden_slow)   # slow -> fast


	def forward(self, x):
		batch_size, n_tokens, n_dim = x.shape
		total_out = torch.zeros(batch_size, n_tokens, n_dim, device=x.device)

		for t in range(n_tokens):                       # one token at a time, so they can talk
			token = x[:, t:t+1]                         # keep the time dim: (B, 1, n_dim)
			perc  = self.perception(token)             # perception reads the token
			fast  = self.fast(self.perc_fast(perc))    # fast reads perception
			slow  = self.slow(self.perc_slow(perc) + self.fast_slow(fast))  # slow reads perc + fast
			# top-down: fast and slow feed back into perception for this step's readout
			total_out[:, t] = self.perception(self.fast_perc(fast) + self.slow_perc(slow))[:, 0]

		return total_out 


class RNN(nn.Module):
	""" Basic RNN block"""

	def __init__(self, n_dim, n_layers, n_hidden, dev, batch_size=None):

		super(RNN, self).__init__()

		self.n_dim     = n_dim
		self.n_hidden  = n_hidden
		self.n_layers  = n_layers
		self.dev       = dev

		self.rnn = nn.GRU(n_dim, n_hidden, n_layers, batch_first=True).to(self.dev)
		self.out  = nn.Linear(n_hidden, n_dim, bias=True).to(self.dev)
		self.sig  = nn.Sigmoid().to(self.dev)

		if batch_size is not None:
			self.init_hidden(batch_size)
		else:
			self.hidden = None

	def init_hidden(self, batch_size):
		h_size = [self.n_layers, batch_size, self.n_hidden]
		self.hidden = torch.zeros(*h_size).to(self.dev)

	def forward(self, x):

		self.init_hidden(x.shape[0])
		x, self.hidden = self.rnn(x, self.hidden)
		# Keeping the outputs off 0 and 1 bounds the BCELoss gradient (see P_EPS)
		x_out  = self.sig(self.out(x)).clamp(P_EPS, 1 - P_EPS)

		return x_out




def loss_function():
	return nn.BCELoss()


def train_network(n_dim=10, n_hidden=32, lr=0.01, n_batches=301, sigma=0.5):
	# n_dim how many "notes" each step of a song has 

	band = songwriter.DiscoBand(n_dim = n_dim) # one world, fixed rules

	# NETWORK
	# the GRU, 32 memory; Bigger batch = smoother but slower learning; 
	# bigger memory = more capacity but slower.
	net = RNN(n_dim=n_dim, n_layers=1, n_hidden=n_hidden, dev='cpu')
	loss_fn = loss_function()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)

	# n_hidden = 32 hidden memory, bigger = more capacity
	# BCELoss (binary cross-entropy) -> binary note, probability between 0 and 1.
	# Optimizer does the learning. Too big -> unstable, Too small -> slow learning


	# Training loop

	for step in range(n_batches): # = N +1 to make sure step N is included

		optimizer.zero_grad()  # forget last step, clear the old corrections

		# clean = noise free songs 
		clean = band.write_album(20, 32) # 32 songs, 20 steps
		clean = torch.tensor(clean, dtype = torch.float32) # numpy

		noisy  = clean + sigma * torch.randn_like(clean) # make noisy copy of the songs (Gaussian); 0.5 is how much noise, we can increase the make the job harder
		output = net(noisy) # network tries to clean it
		loss   = loss_fn(output, clean) # prints out a number saying how wrong that attempt was

		loss.backward() # work out which way to manipulate every weight/figure out the new nudge
		optimizer.step() # take the manipulation/apply the nudge


		if step % 50 == 0:
			print(f"step {step:3d} loss {loss.item():.4f}")
