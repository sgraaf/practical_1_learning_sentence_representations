from data import *

print('Loading the SNLI dataset(s)... ', end='')
train, dev, test = snli()
print('Done!')

print(len(train))
print(train[:10])

