import numpy as np
import timeit, csv
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
os.chdir('tpch-dbgen')
lineitem = np.genfromtxt("lineitem.tbl",delimiter='|', names=["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"],dtyp$
def q6_computation(filter_shipdate,discount,quantity,extendedprice):
#  filter_shipdate = True #tf.not_equal(filter_shipdate,0)
  aux = tf.constant([0.05],dtype=tf.float32)
  filter_3 = tf.greater_equal(discount,aux)
  aux = tf.constant([0.07],dtype=tf.float32)
  filter_4 = tf.less_equal(discount,aux)
  aux = tf.constant([24],dtype=tf.float32)
  filter_5 = tf.less_equal(quantity,aux)
  complete_filter = tf.logical_and(filter_shipdate,tf.logical_and(filter_3,tf.logical_and(filter_4,filter_5)))
  extendedprice = tf.boolean_mask(extendedprice,complete_filter)
#  quantity = tf.boolean_mask(quantity,complete_filter)
  c = tf.multiply(extendedprice, quantity)
  d = tf.reduce_sum(c)
def q6(): 
  filter_1 = lineitem['l_shipdate'] >="1994-01-01"
  filter_2 = lineitem['l_shipdate'] < "1995-01-01"
 # inputs = [ tf.constant(filter_1&filter_2,dtype=tf.bool), tf.constant(lineitem['l_discount'],dtype=tf.float64), tf.constant(lineitem['l_quantity'],dtype=tf.float64), tf.constant(lineitem['l_extendedprice'],dtype=tf.float64)]
  #inputs = [tf.constant(lineitem['l_quantity'],dtype=tf.float64), tf.constant(lineitem['l_extendedprice'],dtype=tf.float64)]
  #inputs = [2.0,5.0]
  inputs = [tf.convert_to_tensor(filter_1 & filter_2, np.bool),tf.convert_to_tensor(lineitem['l_discount'], np.float32),tf.convert_to_tensor(lineitem['l_quantity'], np.float32),tf.convert_to_tensor(lineitem['l_quantity'], np.float32)]
  tpu_computation = tpu.rewrite(q6_computation, inputs)
  tpu_grpc_url = TPUClusterResolver(
    tpu=[os.environ['TPU_NAME']]).get_master()
  with tf.Session(tpu_grpc_url) as sess:
    sess.run(tpu.initialize_system())
    sess.run(tf.global_variables_initializer())
    res = sess.run(tpu_computation)
    sess.run(tpu.shutdown_system())
  return res
n = 5
start_time = timeit.default_timer()
q6()
print(timeit.default_timer() - start_time)
#def bench(q):
#  res = np.median(timeit.repeat("q6()", number=1, repeat=n))
#  print(res)
#bench(6)
