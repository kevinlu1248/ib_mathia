import json
import time
import sys
import csv
from queue import Queue
from typing import Tuple

from tqdm import tqdm

csv.field_size_limit(sys.maxsize)
N = 1998770

print("Loading graph...")
with open("data/dag.csv", "r") as f:
    g = [[int(i) for i in row.split()] for index, row in tqdm(csv.reader(f), total=N)]
print("Done loading graph.")

results = []
t = 0.2  # constant bound for error
m = 1000  # max cycle length we are counting

if __name__ == "__main__":
    for vertex, edges in enumerate(g):
        if vertex in edges:
            nlst = g[vertex]
            nlst.remove(vertex)
            g[vertex] = nlst
            print(f"{vertex} found in its edges")
    writer = csv.writer(open("data/dag_clean.csv", "w"))
    for vertex, edges in enumerate(g):
        writer.writerow((vertex, " ".join(map(str, edges))))
