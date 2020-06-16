from minerva2 import Minerva2
import numpy as np

FEATURES_PER_TRACE = 8

model = Minerva2(FEATURES_PER_TRACE)

for _ in range(10):
    rand_trace = np.random.randint(-1, 2, FEATURES_PER_TRACE)
    model.add_trace(rand_trace)
    
probe = np.random.randint(-1, 2, FEATURES_PER_TRACE)
echo_intensity = model.get_echo_intensity(probe, return_all=False)
print('get_echo_intensity returned', echo_intensity)

model.pretty_print_echo_intensity(probe)