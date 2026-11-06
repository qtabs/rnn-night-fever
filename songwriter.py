
import numpy as np
import itertools

class DiscoBand():
    """ Basic stochastic generator """

    def __init__(self, n_dim, seed=None):

        super(DiscoBand, self).__init__()

        self.n_dim   = n_dim #this is the length of the vectors? 
        self.seed    = seed
        self.rng     = np.random.default_rng(seed)  # use self.rng for all sampling

        #for each band, these things are set: 
        self.harmony = self.sample_harmonic_rules() #super structure WEIGHTS 
        self.motifs, self.super_to_sub = self.sample_disco_motifs () #sub structure WEIGHTS and super structure states to sub structure system mappings 
        self.subsystem_state_vectors = self.sample_state_vectors() #sub structure states to vectors mappings 
                                                                   #and the vectors change for every band... 

    def sample_state_vectors(self):
        # SUBSYSTEM STATEs TO VECTORS DICTIONARY, FIXED PER BAND, SAVED AS subsystem_state_vectors 

        subsystem_states = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
            'B1', 'B2', 'B3', 'B4', 'B5',
            'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
            'D1', 'D2', 'D3', 'D4', 'D5',
            'E1', 'E2', 'E3', 'E4', 'E5', 'E6',
        ]

        all_vectors = np.array(list(itertools.product([0, 1], repeat=self.n_dim)))
        # every ordered combination of 0s and 1s of length n_dim
        # shape (64, 6)
        pick = self.rng.permutation(len(all_vectors))[:len(subsystem_states)]
        # shuffle indexes, keep first 28 indexes

        return dict(zip(subsystem_states, all_vectors[pick].tolist()))
        #retrieve vectors corresponding to first 28 indexes


    def sample_harmonic_rules(self):
        #SUPER STRUCTURE OF THE BAND (TRANS PROBS MATRIX), SAVED AS transitions 

        # Stochastically chooses the super-structure of the generative model.
        # Outputs a dictionary (harmony) containing all the parameters of the superstructure.
        # Note: use self.rng.XXX instead of np.random.XXX for sampling

        a, b    = self.rng.dirichlet(np.ones(2))
        c, d = self.rng.dirichlet(np.ones(2))
        transitions = np.array([
            #     1     2     3     4
            [  0.0,    a,    b,  0.0],  # 1
            [  0.0,  0.0,  0.0,  1.0],  # 2
            [  0.0,    c,  0.0,    d],  # 3
            [  1.0,  0.0,  0.0,  0.0],  # 4
        ])

        return transitions

    def sample_disco_motifs(self):
        # SUBSYSTEMS A B C D E (TRANS PROBS MATRIX) saved as motifs dict 
        # RANDOMISES MAPPING BETWEEN SUBSYSTEM STATES A B C D E and SUPER STRUCTURE STATES 1 2 3 4 5 
        # saved as super_to_sub dict 

        # Stochastically chooses the mini-structure of the generative model A-E 
        # Outputs a list, called motifs, containing the stochastic parameters of the motifs.
        # Note: use self.rng.XXX instead of np.random.XXX for sampling

        # Each ends in an absorbing state (all-zero row) = the motif is over.

        split = lambda n: self.rng.dirichlet(np.ones(n))  # n random weights summing to 1
        a1, a2     = split(2)
        b1, b2     = split(2)
        b3, b4     = split (2)
        c1, c2     = split(2)
        d1, d2, d3 = split(3)
        d4, d5     = split(2)
        e1, e2, e3 = split(3)
         
        motifs = {
            'A': np.array([
                #    1    2    3    4    5    6
                [  0.0,  a1,  a2, 0.0, 0.0, 0.0],  # 1
                [  0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 2
                [  0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 3
                [  0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 4
                [  0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 5
                [  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 6
            ]),
            'B': np.array([
                #    1    2    3    4    5
                [  0.0,  b1,  b2, 0.0, 0.0],  # 1
                [  0.0, 0.0, 0.0, 0.0, 1.0],  # 2
                [  0.0, b3, 0.0, b4, 0.0],  # 3
                [  0.0, 0.0, 1.0, 0.0, 0.0],  # 4
                [  0.0, 0.0, 0.0, 0.0, 0.0],  # 5
            ]),
            'C': np.array([
                #    1    2    3    4    5    6
                [  0.0, 1,  0.0,  0.0, 0.0, 0.0],  # 1
                [  0.0, 0.0, c1, c2, 0.0, 0.0],  # 2
                [  0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 3
                [  0.0, 0.0, 1, 0.0, 0.0, 0.0],  # 4
                [  0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 5
                [  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 6
            ]),
            'D': np.array([
                #    1    2    3    4    5
                [  0.0,  d1,  d2,  d3, 0.0],  # 1
                [  0.0, 0.0,  d4, 0.0,  d5],  # 2
                [  0.0, 0.0, 0.0, 0.0, 1.0],  # 3
                [  0.0, 0.0, 0.0, 0.0, 1.0],  # 4
                [  0.0, 0.0, 0.0, 0.0, 0.0],  # 5
            ]),
            'E': np.array([
                #    1    2    3    4    5    6
                [  0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 1
                [  0.0, 0.0,  e1,  e2,  e3, 0.0],  # 2
                [  0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 3
                [  0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 4
                [  0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 5
                [  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 6
            ]),
        }

        # randomise [A, B, C, D, E], then map macro states 1-5 onto it
        letters = ['A', 'B', 'C', 'D', 'E']
        self.rng.shuffle(letters)
        super_to_sub = {1: letters[0], 2: letters[1], 3: letters[2], 4: letters[3]} 
        return motifs, super_to_sub

    def write_song(self, n_tokens):
        # Generates a stochastic sequence of observations of length n_tokens
        # Outputs a numpy array of dimensions (1, n_tokens, n_dim)

        # STEP 1: sample the super sequence from the super TPM.
        current = 0                                       # always start row 0 (state 1)
        super_sequence = [current]                        #where super sequence is saved 

        while len(super_sequence) < n_tokens: # more than enough 
            row     = self.harmony[current]           # the row for where I am now
            current = self.rng.choice(len(row), p=row)    # choose next state according to prob distribution of row 
            super_sequence.append(current)                # append to super_sequence 
                                                          #super_sequence is cut off when reach n_tokens 

        # STEP 2: map each super state number onto its subsystem letter
        letter_sequence = [self.super_to_sub[s + 1] for s in super_sequence]   # +1: dict keyed 1-5, indices 0-4
        # letter_sequence now looks like ['A', 'C', 'D', 'A', ...]

        # STEP 3: expand each letter into a walk through that subsystem's own TPM
        song_subsystem_states = []
        for letter in letter_sequence:
            tpm     = self.motifs[letter]                            # this subsystem's TPM
            current = 0                                              # start in state 1 (index 0)
            song_subsystem_states.append(f"{letter}1")               # write down where we started

            while tpm[current].sum() > 0:                            # while the row is not all zeroes (i.e. it is not an absorbing state)
                row     = tpm[current]                               # the row for where we are now
                current = self.rng.choice(len(row), p=row)           # <-- sample from probs distribution for next state 
                song_subsystem_states.append(f"{letter}{current + 1}")  #append the state 
        # song_subsystem_states now looks like ['A1','A2','A4','A5','A6','C1', ...]

        # STEP 4: cut to exactly n_tokens, then swap each label for its binary vector
        song_subsystem_states = song_subsystem_states[:n_tokens]
        song = np.array([self.subsystem_state_vectors[s] for s in song_subsystem_states])
        return song.reshape(1, n_tokens, self.n_dim) 


    def write_album(self, n_tokens, n_songs):
        # Generates a stochastic batch (n_songs batches) of sequences (n_tokens length)
        # Outputs a numpy array of dimensions (n_songs, n_tokens, n_dim)

        return np.concatenate([self.write_song(n_tokens) for _ in range(n_songs)], axis=0)

class DiscoFirm():
    def __init__(self, n_dim, n_bands, noise_sigma=0.1, seed=None):                    # noise_sigma = the sigma in y = x + sigma*epsilon; seed=None -> fresh OS entropy
        *bs, ns = np.random.SeedSequence(seed).spawn(n_bands + 1)                      # one independent child seed per band (bs) + one spare for noise (ns)
        self.bands = [DiscoBand(n_dim, s) for s in bs]                                 # each band = its own TPMs, its own 1-4 -> motif map, its own vector dictionary
        self.noise_rng = np.random.default_rng(ns)
        self.noise_sigma = noise_sigma                                                 # noise on a separate stream, so drawing it never perturbs the bands' data

    def add_noise(self, x):
        nu = self.noise_rng.standard_normal(x.shape)
        return x + self.noise_sigma * abs(nu)          # epsilon ~ N(0,1) per element; new float array, x untouched; advances noise_rng each call

    def generate_discography(self, n_tokens, n_songs):
        d = np.concatenate([b.write_album(n_tokens, n_songs) for b in self.bands])     # (n_bands*n_songs, n_tokens, n_dim), blocked by band: rows 0..n_songs-1 are band 0
        return d                                   
    
    def generate_sample(self, n_tokens, n_songs):
        """ Returns (clean (d), noisy (y)) """
        d = self.generate_discography(n_tokens, n_songs)
        y = self.add_noise(d)
        return d, y



## USAGE
## 
## Use the following code to generate training sequences:
##    Start a band x, then 
##    x = DiscoFirm (n_dim=?, n_bands=?)
##    d, y = x.generate_sample (n_tokens=?, n_songs=?)
##    
##    d is the clean sample
##    y is the sample with noise 
## 
##    replace ? with parameters you want: 
## 
##    n_dim is how long the vectors will be 
##    n_bands is how many bands 
##    n_tokens is how long each song will be 
##    n_songs is how many songs 
## 
##    output shape: (n_bands * n_songs, n_tokens, n_dim)
## 
## 
## EXAMPLE: 
## motown = DiscoFirm(n_dim=6, n_bands=5)
## motown.generate_sample(n_tokens=10, n_songs=2)
## 