import numpy as np
import matplotlib.pyplot as plt

n = 16
v = n*n
h = 4*v
ex_h = 1.0

v_state = np.random.choice([-1,1], v)
h_state = np.random.choice([-1,1], h)

a = np.random.random(v)*2-1
b = np.random.random(h)*2-1
w = np.random.uniform(
			low=-0.01 * np.sqrt(6. / (h + v)),
                       	high=0.01 * np.sqrt(6. / (h + v)),
                       	size=(h, v)).astype('complex128')
w += 1.0j*np.random.uniform(
			low=-0.01 * np.sqrt(6. / (h + v)),
                       	high=0.01 * np.sqrt(6. / (h + v)),
                       	size=(h, v))


def psi(v_state):
    return np.exp(np.einsum('i,i->', a, v_state))*np.prod(np.cosh(b+np.einsum('ij,j->i', w, v_state)))


def E(v_state, h_state):
    return np.einsum('i,i->', v_state, a) + np.einsum('i,i->', h_state, b) + \
        np.einsum('i,ij,j->',h_state, w, v_state)

E(v_state, h_state)

def neighbor(v_state, index):
    i = index // n
    j = index % n
    return v_state[index] * (v_state[(i-1)%n*n+j]+v_state[(i+1)%n*n+j]+v_state[i*n+(j+1)%10]+v_state[i*n+(j-1)%10])


def Hamiltonian(v_state):
    return -ex_h*np.einsum('i->', v_state) - np.sum([neighbor(v_state,i) for i in range(v)])

Hamiltonian(v_state)

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def p_hi_given_v(i, v_state):
    return sigmoid(b[i]+np.einsum('ij,j->', w, v_state))

def p_vi_given_h(i, h_state):
    return sigmoid(a[i]+np.einsum('ij,i->', w, h_state))
    


def sample_h_given_v(v_state):
    r = np.random.random(h)
    p = [p_hi_given_v(i, v_state) for i in range(h)]
    return (r<p).astype('int')

def sample_v_given_h(h_state):
    r = np.random.random(v)
    p = [p_vi_given_h(i, h_state) for i in range(v)]
    return (r<p).astype('int')


def f():
    h_state = np.random.randint(0,2,h)
    return Hamiltonian(sample_v_given_h(h_state))

np.mean([f() for i in range(100)])
