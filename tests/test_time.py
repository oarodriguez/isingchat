import time

start = time.perf_counter()
# all = range(3000)
for i in range(3000):
    i+1
end = time.perf_counter()
elapsed = end - start
print(elapsed)
