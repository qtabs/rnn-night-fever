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

	def __init__(self, n_dim, dev, sizes=None, batch_size=None, n_msg=None):

		super(NightFever, self).__init__()

		if sizes is None:
			n_hidden_perception = 2 * n_dim   # P smallest
			n_hidden_fast = 4 * n_dim         # H small, limited capacity
			n_hidden_slow = 32 * n_dim        # C much larger
		else:
			n_hidden_perception, n_hidden_slow, n_hidden_fast = sizes

		self.n_p = n_hidden_perception
		self.n_c = n_hidden_slow
		self.n_h = n_hidden_fast
		self.n_dim = n_dim
		self.dev   = dev
		self.batch_size = batch_size
		self.n_msg = n_dim if n_msg is None else n_msg    # bottleneck size

		# three networks, one step at a time
		self.cell_p = nn.GRUCell(n_dim + self.n_msg, self.n_p)   # input: item + message
		self.cell_h = nn.GRUCell(self.n_msg, self.n_h)           # input: message
		self.cell_c = nn.GRUCell(self.n_msg, self.n_c)           # input: message

		# the six links, sender's state > bottleneck msg
		self.p_to_h = nn.Linear(self.n_p, self.n_msg)   # perception -> fast
		self.p_to_c = nn.Linear(self.n_p, self.n_msg)   # perception -> slow
		self.h_to_c = nn.Linear(self.n_h, self.n_msg)  # fast -> slow
		self.h_to_p = nn.Linear(self.n_h, self.n_msg)   # fast -> perception
		self.c_to_p = nn.Linear(self.n_c, self.n_msg)  # slow -> perception
		self.c_to_h = nn.Linear(self.n_c, self.n_msg)   # slow -> fast
		
		# only P produces the output
		self.readout = nn.Linear(self.n_p, n_dim)

		self.to(dev)

	def forward(self, x):
		batch_size, n_tokens, n_dim = x.shape

		# memories are reset once per episode
		p = torch.zeros(batch_size, self.n_p, device=x.device)
		h = torch.zeros(batch_size, self.n_h, device=x.device)
		c = torch.zeros(batch_size, self.n_c, device=x.device)
		
		keys = ["p", "h", "c",
		        "msg_p_to_h", "msg_p_to_c", "msg_h_to_p",
		        "msg_h_to_c", "msg_c_to_p", "msg_c_to_h"]
		record = {k: [] for k in keys}
		ys = []

		for t in range(n_tokens):
			# messages from previous step's states
			m_p_to_h = self.p_to_h(p)
			m_p_to_c = self.p_to_c(p)
			m_h_to_p = self.h_to_p(h)
			m_h_to_c = self.h_to_c(h)
			m_c_to_p = self.c_to_p(c)
			m_c_to_h = self.c_to_h(c)

			# each networks input (sum of two messages it receives)
			# P receives addtional item
			in_p = torch.cat([x[:, t], m_h_to_p + m_c_to_p], dim=-1)   # item + message
			in_h = m_p_to_h + m_c_to_h
			in_c = m_p_to_c + m_h_to_c

			# everyone updates
			p = self.cell_p(in_p, p)
			h = self.cell_h(in_h, h)
			c = self.cell_c(in_c, c)

			# P produces output
			ys.append(torch.sigmoid(self.readout(p)).clamp(P_EPS, 1 - P_EPS))

			for k, v in zip(keys, [p, h, c, m_p_to_h, m_p_to_c, m_h_to_p,
			                       m_h_to_c, m_c_to_p, m_c_to_h]):
				record[k].append(v)
				
		y = torch.stack(ys, dim=1)
		states = {k: torch.stack(v, dim=1) for k, v in record.items()}
		return y, states


		 


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
