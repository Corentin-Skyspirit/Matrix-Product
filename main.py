import matplotlib.pyplot as plt
import numpy as np
import json

f = json.load(open("out.json", "r"))
for k in f:
    print(k, ":", f[k])
