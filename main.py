N, M = map(int, input().split())
A = list(map(int, input().split()))
A.sort()
sum_list = []

tmp_sum = A[0]
# print(A)

for i in range(N-1):
    if A[i] == A[i+1] or A[i] + 1 == A[i+1]:
        tmp_sum += A[i+1]
    else:
        sum_list.append(tmp_sum)
        tmp_sum = A[i+1]
        
sum_list.append(tmp_sum)

if (A[-1] + 1) % M == A[0]:
    sum_list.append(sum_list[0] + sum_list[-1])

print(max(0, sum(A) - max(sum_list)))
    
