from collections import deque
import random

class Memory:

    def __init__(self,memory_length):
        self.memory = deque()
        self.memory_length = memory_length

    def remember(self,sample):
        if len(self.memory)==self.memory_length:
            self.memory.popleft()
        self.memory.append(sample)

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def memorylength(self):
        return len(self.memory)