import os
import pickle

start_from = 'save'
if os.path.isfile(os.path.join(start_from, 'histories_.pkl')):
    with open(os.path.join(start_from, 'histories_.pkl'), 'rb') as f:
        histories = pickle.load(f)
