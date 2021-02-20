from typing import List,Dict
from DataStructures.Node import Node
import numpy as np

def tournament_selection(population:List,
						 tournament_size:int,
						 survivors:float)->List:
	"""Execute simple tournament selection on the population without 
	replacement. Selects the best member until it reaches the desired remaining
	members in each cluster.
	Parameters:
		tournament_size (int): The number of members selected for each 
			tournament.
		survivors (float): The percent of each cluster that should survive into
			the next generation.
	Returns the surviving members."""
	cluster_count = max(population,key=lambda k: k["cluster"])["cluster"]
	cluster_count += 1

	survivors = int(len(population)*survivors)

	next_generation = []
	while len(next_generation) < survivors:
		for cid in range(cluster_count):
			cluster = [pop for pop in population if pop["cluster"] == cid]
			if len(cluster) == 0:
				continue

			contestants = np.random.choice(cluster,tournament_size)
			winner = max(contestants,key=lambda c: c["fitness"])
			population.remove(winner)
			next_generation.append(winner)

			if len(next_generation) >= survivors:
				return next_generation

	return next_generation


def tournament_selection2(population:List,
						 tournament_size:int,
						 survivors:float)->List:
	"""Execute simple tournament selection on the population without 
	replacement. Selects the best member until it reaches the desired remaining
	members in each cluster.
	Parameters:
		tournament_size (int): The number of members selected for each 
			tournament.
		survivors (float): The percent of each cluster that should survive into
			the next generation.
	Returns the surviving members."""
	cluster_count = max(population,key=lambda k: k["cluster"])["cluster"]
	cluster_count += 1

	# survivors = int(len(population)*survivors)

	next_generation = []
	for cid in range(cluster_count):
		cluster_next_gen = []
		cluster = [pop for pop in population if pop["cluster"] == cid]
		
		cluster_survivors = np.ceil(len(cluster)*survivors)

		while len(cluster_next_gen) < cluster_survivors:
			if len(cluster) == 0:
				break

			contestants = np.random.choice(cluster,tournament_size)
			winner = max(contestants,key=lambda c: c["fitness"])
			cluster.remove(winner)
			cluster_next_gen.append(winner)

		next_generation.append(cluster_next_gen)

	return [p for cluster in next_generation for p in cluster]


def main():
	population = 30
	clusters = 4
	pops = []
	for i in range(population):
		pops.append({"popid":i,"fitness":i*14,"cluster":np.random.randint(0,clusters)})

	print("Showing clusters.")
	for c in range(clusters):
		subset = [p for p in pops if p["cluster"]==c]
		print("\t---------------")
		for p in subset:
			print("\t" + str(p))
	print("END Showing clusters.")

	# next_gen = tournament_selection(pops,3,0.25)
	# next_gen = structural_similarity_clustering(pops)

	print("\n After Selection.")
	for c in range(clusters):
		subset = [p for p in next_gen if p["cluster"]==c]
		print("\t---------------")
		for p in subset:
			print("\t" + str(p))
	print("END Showing clusters.")

if __name__ == "__main__":
	main()