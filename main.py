import matplotlib.pyplot as plt
import numpy as np
import sys

nbCPU = []
duree = []

leftright = "right"

f = open("out.txt", "r")
for ligne in f:
    parties = ligne.strip().split(",")
    nbCPU.append(int(parties[0]))
    duree.append(int(parties[1]))

plt.figure(figsize=(8, 5))
plt.plot(nbCPU, duree, marker='o', linestyle='-')
plt.xlabel("Nombre de coeurs")
plt.ylabel("Temps (ns)")
plt.title("Strong scaling avec un layout " + leftright)
plt.grid(True)
plt.tight_layout()
plt.ylim(bottom=0)
plt.savefig("strong_scaling_layout_" + leftright + ".png")

flops = []
args = sys.argv[1:]
for d in duree:
    flops.append((2 * int(args[0]) * int(args[1]) * int(args[2])) / d)

plt.figure(figsize=(8, 5))
plt.plot(nbCPU, flops, marker='o', linestyle='-')
plt.xlabel("Nombre de coeurs")
plt.ylabel("GFlops / s")
plt.title("Strong scaling avec un layout " + leftright)
plt.grid(True)
plt.tight_layout()
plt.ylim(bottom=0)
plt.savefig("strong_scaling_flops_layout_" + leftright + ".png")
