import pulp as pp
import math
import matplotlib.pyplot as plt
import networkx as nx

num = 50
directory = "/Users/hs/MyGraduation/dataset_tsp/tsp_dataset_{0}.csv".format(num)
tsp_list = [[0, 0]]

with open(directory, 'r') as file:
    name1 = file.readline()
    cus_num = int(file.readline())
    name2 = file.readline()
    for i in range(cus_num):
        point = file.readline()
        lista = point.split(",")
        listb = list(map(float,lista))
        tsp_list.append(listb)

n = len(tsp_list)
# print("{0}:{1}".format(name1,cus_num))
# print("{0}:{1}".format(name2,tsp_list))

model_name = "tsp_model{0}".format(num)
model = pp.LpProblem(name = model_name, sense = 1)
x, d, u ={}, {}, {}
for i in range(n):
  u[i] = pp.LpVariable("u(%s)"%(i), cat = "Integer")
  for j in range(n):
    if i != j:
      x[i,j] = pp.LpVariable("x(%s,%s)"%(i,j), lowBound = 0,upBound = 1,cat = 'Binary')
      d[i,j] = round(math.sqrt((tsp_list[i][0] - tsp_list[j][0]) ** 2 + (tsp_list[i][1] - tsp_list[j][1]) ** 2), 4)

model.setObjective(obj = pp.lpSum(d[i,j] * x[i,j] for i in range(n) for j in range(n) if i != j))

for i in range(n):
  model += pp.lpSum(x[i,j] for j in range(n) if i != j) == 1

for i in range(n):
  model += pp.lpSum(x[j,i] for j in range(n) if i != j) == 1


for i in range(1, n):
  for j in range(1, n):
    if i != j:
      model += u[i] - u[j] + (n - 1 + 1) * x[i,j] <= n - 1 

for i in range(1, n):
  model += u[i] >= 1
  model += u[i] <= n - 1

model += u[0] == 0

# model.solve()

model.writeLP(filename = "{0}.lp".format(model_name))

G = nx.DiGraph()
for i in range(n):
  G.add_node((tsp_list[i][0], tsp_list[i][1]))

for i in range(n):
  for j in range(n):
    if i != j:
      if x[i, j].value() == 1:
        G.add_edge((tsp_list[i][0], tsp_list[i][1]),(tsp_list[j][0], tsp_list[j][1]))

pos = {n: (n[1], n[0]) for n in G.nodes()}
node_size = [30 for i in range(n)]
nx.draw_networkx_nodes(G, pos, node_size = node_size, node_color="blue")
nx.draw_networkx_edges(G, pos, edge_color="red")

plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.tick_params(labelbottom = True, labelleft = True)
plt.xticks([n for n in range(-10, 11, 2)])
plt.yticks([n for n in range(-10, 11, 2)])
plt.savefig('{0}.png'.format(model_name))
