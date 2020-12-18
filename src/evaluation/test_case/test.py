# a = []
# node = [1,2]
# for i in node:
#     a.extend([i] * 5)
# print(a)
#
# b = {0: [[0,1,1], [0,2,1], [0,3,1], [0]]}
# print(b[0][1:])
#
# node_neighbor = list(b[0][1:])
#
# if [0] in node_neighbor:
#     node_neighbor.remove([0])
#
# print(node_neighbor)
#
c = {}
roots = [1,2,3]
for root in roots:
    c[root] = {}
    c[root][root] = [root]
    print("!: ", c)
    c[root][root].append(2)
    c[root][2] = [root]
    # print(c[root][1:])
print(c)
# # print(c[1][1:])
#
# d = {}
# nodes = [4]
# for node in nodes:
#     d[node] = {}
#     d[node][node] = [node]
# print(d[nodes[0]])
#
# e = {0:[]}
# print(e[0][1:])

# for i in range(2, 10, 5):
#     print(i)