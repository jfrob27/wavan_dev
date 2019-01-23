import numpy as np

def find_chains(wt):
	'''
	Find 'chains' along zero phase lines
	'''
	
	maxp = 0.1
	
	M = wt.shape[0]
	na = wt.shape[1]
	nb = wt. shape[2]
	
	phase = np.abs(np.arctan(wt.imag/wt.real))
	chains = np.zeros(wt.shape)
	srnd = np.zeros((8))
	
	for j in range(M):
		
		idx = np.where((phase == np.min(phase)) & (phase < maxp))
		#chains[idx] = 1
		phase[idx] = 1
		
		srnd[0] = phase[idx[0]+1, idx[10]]
		
		