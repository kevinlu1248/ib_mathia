import json
import random
import time
import sys
import csv
from queue import Queue
from typing import Tuple, List

from tqdm import tqdm

csv.field_size_limit(sys.maxsize)
N = 1998770
MAX_SELECTION_SIZE = 1000
random.seed(0)

print("Loading graph...")
with open("data/dag_clean.csv", "r") as f:
    g = tuple(
        [int(i) for i in row.split()] for index, row in tqdm(csv.reader(f), total=N)
    )
print("Done loading graph.")

def subgraph(subset):
    sg = [None for _ in range(N)]
    for vertex in subset:
        edges = []
        for outbound in g[vertex]:
            if outbound in subset:
                edges.append(outbound)
        sg[vertex] = edges
    return sg


def bfs():
    all_vertices = set()
    not_sources = set()
    for vertex, edges in enumerate(g):
        if edges is not None:
            all_vertices.add(vertex)
            not_sources.update(edges)
    sources = all_vertices.difference(not_sources)

    selected = set()
    renamer = {}
    cnt = 0

    while len(selected) < MAX_SELECTION_SIZE:
        selected = set()
        renamer = {}
        cnt = 0

        selected_vertex = random.choice(list(sources))

        q = Queue()
        q.put(selected_vertex)

        while not q.empty() and len(selected) < MAX_SELECTION_SIZE:
            vertex = q.get()
            if vertex in selected:
                continue
            selected.add(vertex)
            cnt += 1
            renamer[vertex] = cnt
            for child in g[vertex]:
                if child not in selected:
                    q.put(child)

        print(len(selected))

    return renamer, selected

if __name__ == "__main__":
    renamer, subset = bfs()
    with open("new_data/dag.csv", "w") as f1, open("new_data/edges.csv", "w") as f2:
        graph = csv.writer(f1)
        edges_file = csv.writer(f2)
        for vertex in subset:
            output_verts = []
            for outbound in g[vertex]:
                if outbound in subset:
                    output_verts.append(renamer[outbound])
                    f2.write(f"{renamer[vertex]} {renamer[outbound]}\n")
            graph.writerow((renamer[vertex], " ".join(map(str, output_verts))))
