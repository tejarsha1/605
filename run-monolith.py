import os
import time

start = time.time()
os.system('python3 train.py 1')
train = time.time()
os.system('python3 test.py')
test = time.time()
print('Training:', train - start, 'seconds')
print('Testing:', test - train, 'seconds')