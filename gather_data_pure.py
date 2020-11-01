import json
import random
import time
import sys
import csv
from queue import Queue
from typing import Tuple, List

from tqdm import tqdm
import pandas as pd

csv.field_size_limit(sys.maxsize)
N = 1000

print("Loading graph...")
with open("new_data/dag.csv", "r") as f:
    g = [[] for _ in range(N + 1)]
    for index, row in csv.reader(f):
        g[int(index)] = list(map(int, row.split()))
print("Done loading graph.")

edge_cnt = 0
for i in range(N):
    edge_cnt += len(g[i])

results = [[] for _ in range(10)]
t = 0.2  # constant bound for error
m = 20  # max cycle length found
random.seed(0)


def cyc(rig: tuple, verbose: bool = False) -> List[int]:  # bulk of the work
    cycles = [0 for _ in range(m)]
    removed_paths = set()
    path_id_cnt = 0
    current_node = 1

    all_vertices = set()
    not_sources = set()
    for vertex, edges in enumerate(rig):
        if edges is not None:
            all_vertices.add(vertex)
            not_sources.update(edges)
    sources = all_vertices.difference(not_sources)

    for start in all_vertices:
        visited = [[] for _ in range(N + 1)]
        q = Queue()
        q.put((start, 0))
        while not q.empty():
            vertex, dist = q.get()
            for child in rig[vertex]:
                for depth in visited[child]:
                    cycles[depth + dist + 1] += 1
                visited[child].append(dist)
                q.put((child, dist + 1))
    return cycles


def compute_cyc(arr: List[int], p):
    sm = 0
    for index, val in enumerate(arr):
        sm += val / p ** index
    return sm


def random_subgraph(p: float):
    rig = [None for _ in range(N)]
    vertices = []  # every node has p chance of survival
    for vertex in range(N):
        if random.random() < p:
            vertices.append(vertex)
    vertices_set = set(vertices)
    for vertex in vertices:
        rig[vertex] = []
        for endpoint in g[vertex]:
            if endpoint in vertices_set:
                rig[vertex].append(endpoint)

    return rig


def approximate(T: int) -> Tuple[int, int]:
    p = (T * t * t + 1) ** (-1.0 / 3)  # maximum p as derived in paper
    num_of_vertices_wanted = round(p * len(g))

    start = time.time()
    rig = random_subgraph(p)  # random induced subgraph
    raw = cyc(rig, p)
    value = {"raw": raw, "approximation": compute_cyc(raw, p)}
    _time = time.time() - start

    return {"value": value, "time": _time}


if __name__ == "__main__":
    df = pd.DataFrame(columns=["T", "p", "approximation", "time", "raw"])
    Ts = 20
    no_of_measurements = 50

    print("Performing initial calculation...")
    for _ in range(1, 11):
        start = time.time()
        correct_val = cyc(g)
        true_val = compute_cyc(correct_val, 1)
        elapsed = time.time() - start
        df = df.append(
            {
                "T": 0,
                "p": 1,
                "approximation": true_val,
                "time": elapsed,
                "raw": correct_val,
            },
            ignore_index=True,
        )
    print("Done")

    pbar = tqdm(total=no_of_measurements, desc="Measurement trial")
    for T in tqdm(
        range(1, Ts + 1), total=Ts, desc="Measurement values"
    ):  # called T since it was called T in the paper
        pbar.reset()
        for measurement in range(no_of_measurements):
            pbar.update(1)
            approx = approximate(T)
            df = df.append(
                {
                    "T": T,
                    "p": (T * t * t + 1) ** (-1.0 / 3),
                    "approximation": approx["value"]["approximation"],
                    "time": approx["time"],
                    "raw": approx["value"]["raw"],
                },
                ignore_index=True,
            )
            df.to_csv("results/raw.csv")

    df.to_csv("results/raw.csv")
