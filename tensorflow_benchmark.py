import numpy as np
import timeit, csv
import os
import tensorflow as tf

os.chdir('tpch-dbgen')

lineitem = np.genfromtxt("lineitem.tbl",delimiter='|', names=["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"],dtype=['int32','int32','int32','int32','float64','float64','float64','float64','S1','S1','S10','S10','S10','S25','S10','S44'])

def q6(): 
  filter_shipdate = tf.placeholder(dtype=tf.bool, shape=(None,))
  discount = tf.placeholder(dtype=tf.float64, shape=(None,))
  quantity = tf.placeholder(dtype=tf.float64, shape=(None,))
  extendedprice = tf.placeholder(dtype=tf.float64, shape=(None,))
  aux = tf.constant([0.05],dtype=tf.float64)
  filter_3 = tf.greater_equal(discount,aux)
  aux = tf.constant([0.07],dtype=tf.float64)
  filter_4 = tf.less_equal(discount,aux)
  aux = tf.constant([24],dtype=tf.float64)
  filter_5 = tf.less_equal(quantity,aux)
  complete_filter = tf.logical_and(filter_shipdate,tf.logical_and(filter_3,tf.logical_and(filter_4,filter_5)))
  extendedprice = tf.boolean_mask(extendedprice,complete_filter)
  quantity = tf.boolean_mask(quantity,complete_filter)
  c = tf.multiply(extendedprice, quantity)
  d = tf.reduce_sum(c)
  with tf.Session() as sess:
    # Tensorflow does not support filters on strings
    filter_1 = lineitem['l_shipdate'] >="1994-01-01"
    filter_2 = lineitem['l_shipdate'] < "1995-01-01"
    res = sess.run(d, feed_dict={
        filter_shipdate: filter_1&filter_2,
        discount: lineitem['l_discount'],
        quantity: lineitem['l_quantity'],
        extendedprice: lineitem['l_extendedprice']
      })
  return res

n = 5

def bench(q):
  res = np.median(timeit.repeat("q%d()" % q, setup="from __main__ import q%d" % q, number=1, repeat=n))
  print(res)

bench(6)