import numpy as np
import timeit, csv
import os

os.chdir('tpch-dbgen')

lineitem = np.genfromtxt("lineitem.tbl",delimiter='|', names=["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"],dtype=['int32','int32','int32','int32','float64','float64','float64','float64','S1','S1','S10','S10','S10','S25','S10','S44'])


def q6(): 
	filter_1 = lineitem['l_shipdate'] >="1994-01-01"
	filter_2 = lineitem['l_shipdate'] < "1995-01-01"
	filter_3 = lineitem['l_discount'] >=0.05
	filter_4 =lineitem['l_discount'] <=0.07
	filter_5 = lineitem['l_quantity'] < 24
	l = lineitem[filter_1&filter_2&filter_3&filter_4&filter_5]
	mult = np.multiply(l['l_extendedprice'], l['l_discount'])
	res = np.sum(mult)
	print(res)
	return res

n = 5

def bench(q):
	res = np.median(timeit.repeat("q%d()" % q, setup="from __main__ import q%d" % q, number=1, repeat=n))
	print(res)

bench(6)