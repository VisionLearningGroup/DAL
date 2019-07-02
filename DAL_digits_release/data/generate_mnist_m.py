import pickle
import scipy.io as sio

with open('mnistm_data.pkl', 'rb') as f:
    data = pickle.load(f)
sio.savemat('mnist_m.mat',data)