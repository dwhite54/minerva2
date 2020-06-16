# [CSU Cleary Memory Lab](http://anne.cleary.colostate.edu/)'s MINERVA2 Python model
This model was developed based on Hintzman's MINERVA2 model for the purpose of performing simulations on how features interact within memory traces to produce echo intensities.
This model was developed by [Kat McNeely-White](http://k.mcneelywhite.com/).

To learn about the theory and mechanisms behind MINERVA2, see [Hintzman's Explanation from 1988](https://www.researchgate.net/publication/232588495_Judgments_of_frequency_and_recognition_memory_in_a_multiple-trace_model).

## Prerequisites
To just use the model, you only need `numpy`. 

In the included notebooks, `matplotlib`, `seaborn`, `scipy`, and `pandas` are also used.

## Installation 
Clone this repository or just download `minerva2.py` individually.

## Usage
Usage of the model should be mostly self-explanatory. See below for a simple example:

First, instantiate the model using the desired features per memory trace
```
import numpy as np
from minerva2 import Minerva2

FEATURES_PER_TRACE = 8

model = Minerva2(FEATURES_PER_TRACE)
```

Then, add memory traces to your model. In this example we are using random noise. 

```
for _ in range(10):
    rand_trace = np.random.randint(-1, 2, FEATURES_PER_TRACE)
    model.add_trace(rand_trace)
```
Note: `add_trace(trace)` expects a NumPy array of shape `(FEATURES_PER_TRACE,)`.

Finally, probe your memory model. Again we are using random noise.

```
probe = np.random.randint(-1, 2, FEATURES_PER_TRACE)
model.get_echo_intensity(probe, return_all=False)
```
Note `get_echo_intensity(probe, return_all=False)` expects a NumPy array. `return_all=True` will also return individual activations and similarity scores used in calculating the final echo intensity.

This calculation can be inspected in detail using the `pretty_print_echo_intensity(probe)`. For example, 
```
model.pretty_print_echo_intensity(probe)
```
prints
```
PROBE: [-1, 0, 1, 1, -1, -1, 0, -1]
TRACE 0: [1, 0, 1, -1, 0, -1, 1, -1] ->  0.125^3 =    0.002
TRACE 1: [0, 1, 0, 0, -1, 1, 0, 0] ->  0.000^3 =    0.000
TRACE 2: [0, 0, -1, 1, -1, 1, 1, -1] ->  0.125^3 =    0.002
TRACE 3: [-1, 0, -1, -1, 0, 1, -1, -1] -> -0.125^3 =   -0.002
TRACE 4: [0, 0, -1, -1, -1, 1, 1, 0] -> -0.250^3 =   -0.016
TRACE 5: [1, 1, 1, -1, 0, 1, 0, 0] -> -0.250^3 =   -0.016
TRACE 6: [-1, 1, -1, 0, -1, -1, 0, -1] ->  0.375^3 =    0.053
TRACE 7: [-1, 1, 0, 1, 0, -1, 0, 1] ->  0.250^3 =    0.016
TRACE 8: [1, -1, 0, 1, -1, 0, -1, 0] ->  0.125^3 =    0.002
TRACE 9: [-1, 0, 1, 0, -1, 0, 0, 1] ->  0.250^3 =    0.016
TRACE 10: [1, 0, -1, 0, 1, 0, -1, -1] -> -0.250^3 =   -0.016
TRACE 11: [1, -1, 0, 1, 1, 1, 1, 1] -> -0.375^3 =   -0.053
--------------------------------------------------------------------------------
                                                                          -0.001
```

These steps can also be performed using `python example.py`.

### Large calculations

To take advantage of NumPy's efficient handling of large arrays, use `model.add_traces(probes)` and `model.get_echo_intensities(traces)` for processing multiple probes and/or traces.