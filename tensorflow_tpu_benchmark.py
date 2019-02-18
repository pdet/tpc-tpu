import numpy as np
import timeit, csv
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
os.chdir('tpch-dbgen')
lineitem = np.genfromtxt("lineitem.tbl",delimiter='|', names=["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"],dtyp$

def to_integer(dt_time):
  dt_time = dt_time.split("-")
  return 10000*int(dt_time[0]) + 100*int(dt_time[1]) + int(dt_time[2])

vfunc = np.vectorize(to_integer)
l_shipdate = vfunc(lineitem['l_shipdate'])

def q6_computation(shipdate,discount,quantity,extendedprice):
  zeros = tf.zeros_like(discount)
  complete_filter = tf.logical_and(tf.greater_equal(discount,0.05),tf.logical_and(tf.less_equal(discount,0.07),tf.logical_and(tf.less(quantity,24),tf.logical_and(tf.less(shipdate,19950101),tf.greater_equal(shipdate,19940101)))))
  result = tf.reduce_sum(tf.multiply(tf.where(complete_filter,extendedprice,zeros), tf.where(complete_filter,discount,zeros)))

def q6(): 
  inputs = [tf.convert_to_tensor(shipdate, np.int32),tf.convert_to_tensor(lineitem['l_discount'], np.float32),tf.convert_to_tensor(lineitem['l_quantity'], np.float32),tf.convert_to_tensor(lineitem['l_quantity'], np.float32)]
  tpu_computation = tpu.rewrite(q6_computation, inputs)
  tpu_grpc_url = TPUClusterResolver(
    tpu=[os.environ['TPU_NAME']]).get_master()
  with tf.Session(tpu_grpc_url) as sess:
    sess.run(tpu.initialize_system())
    sess.run(tf.global_variables_initializer())
    res = sess.run(tpu_computation)
    sess.run(tpu.shutdown_system())
  return res

start_time = timeit.default_timer()
q6()
print(timeit.default_timer() - start_time)
