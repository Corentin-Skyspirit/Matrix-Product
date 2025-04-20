import matplotlib.pyplot as plt
import numpy as np
import sys

nbCPU = []
duree = []

f = open("out.txt", "r")
for ligne in f:
    parties = ligne.strip().split(",")
    nbCPU.append(int(parties[0]))
    duree.append(int(parties[1]))

nbCPU=nbCPU[6:]

plt.figure(figsize=(8, 5))
plt.plot(nbCPU, duree[:6], marker='o', linestyle='-', label='Layout Left')
plt.plot(nbCPU, duree[6:], marker='o', linestyle='-', label='Layout Right')
plt.xlabel("Nombre de coeurs")
plt.ylabel("Temps (ns)")
plt.title("Comparaison des layout left et right")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.ylim(bottom=0)
plt.savefig("strong_scaling.png")

flops = []
args = sys.argv[1:]
for d in duree:
    flops.append((2 * int(args[0]) * int(args[1]) * int(args[2])) / d)

plt.figure(figsize=(8, 5))
plt.plot(nbCPU, flops[:6], marker='o', linestyle='-', label='Layout Left')
plt.plot(nbCPU, flops[6:], marker='o', linestyle='-', label='Layout Right')
plt.xlabel("Nombre de coeurs")
plt.ylabel("GFlops / s")
plt.title("Comparaison des layout left et right")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.ylim(bottom=0)
plt.savefig("strong_scaling_flops.png")
