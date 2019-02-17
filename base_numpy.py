import numpy as np
import timeit

column_1 = np.random.uniform(size=100000000)
column_2 = np.random.uniform(size=100000000)

start_time = timeit.default_timer()
a = np.sum(np.multiply(column_1, column_2))
print(timeit.default_timer() - start_time)