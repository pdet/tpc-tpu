import numpy as np
import timeit, csv
import os
import tensorflow as tf

os.chdir('tpch-dbgen')

lineitem = np.genfromtxt("lineitem.tbl",delimiter='|', names=["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"],dtype=['int32','int32','int32','int32','float64','float64','float64','float64','S1','S1','S10','S10','S10','S25','S10','S44'])

def to_integer(dt_time):
  dt_time = dt_time.split("-")
  return 10000*int(dt_time[0]) + 100*int(dt_time[1]) + int(dt_time[2])

vfunc = np.vectorize(to_integer)
l_shipdate = vfunc(lineitem['l_shipdate'])

def q6(): 
  shipdate = tf.placeholder(dtype=tf.int32, shape=(None,))
  discount = tf.placeholder(dtype=tf.float64, shape=(None,))
  quantity = tf.placeholder(dtype=tf.float64, shape=(None,))
  extendedprice = tf.placeholder(dtype=tf.float64, shape=(None,))
  zeros = tf.zeros_like(discount)
  complete_filter = tf.logical_and(tf.greater_equal(discount,0.05),tf.logical_and(tf.less_equal(discount,0.07),tf.logical_and(tf.less(quantity,24),tf.logical_and(tf.less(shipdate,19950101),tf.greater_equal(shipdate,19940101)))))
  result = tf.reduce_sum(tf.multiply(tf.where(complete_filter,extendedprice,zeros), tf.where(complete_filter,discount,zeros)))
  with tf.Session() as sess:
    res = sess.run(result, feed_dict={
        shipdate: l_shipdate,
        discount: lineitem['l_discount'],
        quantity: lineitem['l_quantity'],
        extendedprice: lineitem['l_extendedprice']
      })
    print res
  return res

n = 5

def bench(q):
  res = np.median(timeit.repeat("q%d()" % q, setup="from __main__ import q%d" % q, number=1, repeat=n))
  print(res)

bench(6)