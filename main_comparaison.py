import matplotlib.pyplot as plt
import numpy as np
import sys

nbCPU = 6
duree = list(range(nbCPU * 6))
x = range(1, nbCPU + 1)

f = open("out.txt", "r")
for ligne in f:
    parties = ligne.strip().split(",")
    cpt = int(parties[0]) - 1
    duree[cpt] = int(parties[1])
    duree[cpt + nbCPU] = int(parties[2])
    duree[cpt + nbCPU * 2] = int(parties[3])
    duree[cpt + nbCPU * 3] = int(parties[4])
    duree[cpt + nbCPU * 4] = int(parties[5])
    duree[cpt + nbCPU * 5] = int(parties[6])

plt.figure(figsize=(8, 5))
plt.plot(x, duree[:6], marker='o', linestyle='-', label='Classic version')
plt.plot(x, duree[6:12], marker='o', linestyle='-', label='Cache blocked version (16)')
plt.plot(x, duree[12:18], marker='o', linestyle='-', label='Cache blocked version (32)')
plt.plot(x, duree[18:24], marker='o', linestyle='-', label='Cache blocked version (64)')
plt.plot(x, duree[24:30], marker='o', linestyle='-', label='Cache blocked version (128)')
plt.plot(x, duree[30:], marker='o', linestyle='-', label='Cache blocked version (256)')
plt.xlabel("Nombre de coeurs")
plt.ylabel("Temps (ns)")
plt.title("Temps d'exécution d'un version avec et sans cache blocking")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.ylim(bottom=0)
plt.savefig("strong_scaling_cache_blocking.png")

flops = []
args = sys.argv[1:]
for d in duree:
    flops.append((2 * int(args[0]) * int(args[1]) * int(args[2])) / d)

plt.figure(figsize=(8, 5))
plt.plot(x, flops[:6], marker='o', linestyle='-', label='Classic version')
plt.plot(x, flops[6:12], marker='o', linestyle='-', label='Cache blocked version (16)')
plt.plot(x, flops[12:18], marker='o', linestyle='-', label='Cache blocked version (32)')
plt.plot(x, flops[18:24], marker='o', linestyle='-', label='Cache blocked version (64)')
plt.plot(x, flops[24:30], marker='o', linestyle='-', label='Cache blocked version (128)')
plt.plot(x, flops[30:], marker='o', linestyle='-', label='Cache blocked version (256)')
plt.xlabel("Nombre de coeurs")
plt.ylabel("GFlops / s")
plt.title("GFlops / s d'une version en cache blocking et une sans")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.ylim(bottom=0)
plt.savefig("strong_scaling_flops_cache_blocking.png")
