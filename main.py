# import collections
# import sys
# from sys import stdin
# import fractions
# from collections import Counter, defaultdict
# from itertools import permutations
# from collections import deque
# import heapq

# import math
# sys.setrecursionlimit(10**8)

# import bisect
# def I():
#     return stdin.readline().rstrip()
# def MI():
#     return map(int,stdin.readline().rstrip().split())
# def LI():
#     return list(map(int,stdin.readline().rstrip().split()))
# inf = 10**60 + 7


# from collections import defaultdict

# class UnionFind():
#     """
#     Union Find木クラス

#     Attributes
#     --------------------
#     n : int
#         要素数
#     root : list
#         木の要素数
#         0未満であればそのノードが根であり、添字の値が要素数
#     rank : list
#         木の深さ
#     """

#     def __init__(self, n):
#         """
#         Parameters
#         ---------------------
#         n : int
#             要素数
#         """
#         self.n = n
#         self.root = [-1]*(n+1)
#         self.rank = [0]*(n+1)

#     def find(self, x):
#         """
#         ノードxの根を見つける

#         Parameters
#         ---------------------
#         x : int
#             見つけるノード

#         Returns
#         ---------------------
#         root : int
#             根のノード
#         """
#         if(self.root[x] < 0):
#             return x
#         else:
#             self.root[x] = self.find(self.root[x])
#             return self.root[x]

#     def unite(self, x, y):
#         """
#         木の併合

#         Parameters
#         ---------------------
#         x : int
#             併合したノード
#         y : int
#             併合したノード
#         """

#         x = self.find(x)
#         y = self.find(y)

#         if(x == y):
#             return
#         elif(self.rank[x] > self.rank[y]):
#             self.root[x] += self.root[y]
#             self.root[y] = x
#         else:
#             self.root[y] += self.root[x]
#             self.root[x] = y
#             if(self.rank[x] == self.rank[y]):
#                 self.rank[y] += 1

#     def same(self, x, y):
#         """
#         同じグループに属するか判定

#         Parameters
#         ---------------------
#         x : int
#             判定したノード
#         y : int
#             判定したノード

#         Returns
#         ---------------------
#         ans : bool
#             同じグループに属しているか
#         """
#         return self.find(x) == self.find(y)

#     def size(self, x):
#         """
#         木のサイズを計算

#         Parameters
#         ---------------------
#         x : int
#             計算したい木のノード

#         Returns
#         ---------------------
#         size : int
#             木のサイズ
#         """
#         return -self.root[self.find(x)]

#     def roots(self):
#         """
#         根のノードを取得

#         Returns
#         ---------------------
#         roots : list
#             根のノード
#         """
#         return [i for i, x in enumerate(self.root) if x < 0]

#     def group_size(self):
#         """
#         グループ数を取得

#         Returns
#         ---------------------
#         size : int
#             グループ数
#         """
#         return len(self.roots())

#     def group_members(self):
#         """
#         全てのグループごとのノードを取得

#         Returns
#         ---------------------
#         group_members : defaultdict
#             根をキーとしたノードのリスト
#         """
#         group_members = defaultdict(list)
#         for member in range(self.n):
#             group_members[self.find(member)].append(member)
#         return group_members


# def main():
#     N, A, B = MI()
#     d_set = set()
#     d_list = LI()
#     for i in range(N):
#         d_set.add(d_list[i] % (A+B))

#     # 1週間のうちの休日は連続しているので，最大値と最小値のみに着目すれば良い
#     d_list = []
#     for d in d_set:
#         d_list.append(d)
#     d_list.sort()
#     # print(d_list)
#     d_list.append(d_list[0])
#     for i in range(len(d_list)-1):
#         length = (d_list[i] - d_list[i+1]) % (A+B)
#         # assert length != 0
#         # print(length)
#         if length + 1 <= A:
#             print("Yes")
#             return 0
#     print("No")

# if __name__ == "__main__":
#     main()

import sys
from collections import defaultdict

def main(lines):
    # このコードは標準入力と標準出力を用いたサンプルコードです。
    # このコードは好きなように編集・削除してもらって構いません。
    # ---
    # This is a sample code to use stdin and stdout.
    # Edit and remove this code as you like.

    n_cnt = defaultdict(int)
    for n in lines[0]:
        n_cnt[int(n)] += 1

    # 0でないものは必ずあるので，最小のものを探す．
    ans = "" # 答え
    for n in range(1, 10):
        if n_cnt[n] > 0:
            ans += str(n)
            n_cnt[n] -= 1
            break

    # 残りは，0も含めて下から順番に入れていく
    for n in range(0, 10):
        ans += str(n) * n_cnt[n]
    print(ans)

if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)
