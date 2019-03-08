import numpy as np
import timeit, csv
import os

os.chdir('tpch-dbgen')

lineitem = np.genfromtxt("lineitem.tbl",delimiter='|', names=["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"],dtype=['int32','int32','int32','int32','float32','float32','float32','float32','S1','S1','S10','S10','S10','S25','S10','S44'])

def to_integer(dt_time):
  dt_time = dt_time.split("-")
  return 10000*int(dt_time[0]) + 100*int(dt_time[1]) + int(dt_time[2])

vfunc = np.vectorize(to_integer)
l_shipdate = vfunc(lineitem['l_shipdate'])

def q6(): 
	filter_1 = l_shipdate >=19940101
	filter_2 = l_shipdate < 19950101
	filter_3 = lineitem['l_discount'] >=0.05
	filter_4 =lineitem['l_discount'] <=0.07
	filter_5 = lineitem['l_quantity'] < 24
	l = lineitem[filter_1&filter_2&filter_3&filter_4&filter_5]
	mult = np.multiply(l['l_extendedprice'], l['l_discount'])
	res = np.sum(mult)
	print(res)
	return res

n = 5

start_time = timeit.default_timer()
q6()
print(timeit.default_timer() - start_time)