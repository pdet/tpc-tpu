import pandas as pd
import numpy as np
import timeit, csv
import os

os.chdir('tpch1')

lineitem = pd.read_csv("lineitem.tbl", sep='|', names=["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"],dtype=['int32','int32','int32','int32','float32','float32','float32','float32','S1','S1','S10','S10','S10','S25','S10','S44'])


def udf_disc_price(extended, discount):
	return np.multiply(extended, np.subtract(1, discount))

def udf_charge(extended, discount, tax):
	return np.multiply(extended, np.multiply(np.subtract(1, discount), np.add(1, tax)))

def q1():
	df = lineitem[["l_shipdate", "l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount", "l_tax"]][(lineitem['l_shipdate'] <= '1998-09-01')]
	df['disc_price'] = udf_disc_price(df['l_extendedprice'], df['l_discount'])
	df['charge']     = udf_charge(df['l_extendedprice'], df['l_discount'], df['l_tax'])
	return df.groupby(['l_returnflag', 'l_linestatus'])\
	  		 .agg({'l_quantity': 'sum', 'l_extendedprice': 'sum', 'disc_price': 'sum', 'charge': 'sum',
					 'l_quantity': 'mean', 'l_extendedprice': 'mean', 'l_discount': 'mean', 'l_shipdate': 'count'})

def q6(): 
	l = lineitem[["l_extendedprice", "l_discount", "l_shipdate", "l_quantity"]][
		(lineitem.l_shipdate >= "1994-01-01") & 
		(lineitem.l_shipdate < "1995-01-01") & 
		(lineitem.l_discount >= 0.05) & 
		(lineitem.l_discount <= 0.07) & 
		(lineitem.l_quantity < 24)][["l_extendedprice", "l_discount"]]
	res = (l.l_extendedprice * l.l_discount).sum()
	print(res)
	return res


##### 
n = 5

f = open("pandas.csv", 'w')
writer = csv.writer(f)


def bench(q):
	res = timeit.repeat("q%d()" % q, setup="from __main__ import q%d" % q, number=1, repeat=n)
	print(res[4])
	writer.writerow(["pandas", "%d" % q, "%f" % res[4]])
	f.flush()

# bench(1)
bench(6)


