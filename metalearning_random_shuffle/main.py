import string

dataset = list(string.ascii_letters + string.digits)
print(dataset)

# copy
print(dataset * 2)

# cut
print(dataset[0:5] + dataset[10:])

# shuffle
import random

random.shuffle(dataset)
print(dataset)

# and this is what you must learn
