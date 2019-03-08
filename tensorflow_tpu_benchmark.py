import numpy as np
import timeit, csv
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.python.client import timeline

os.chdir('tpctpu')
lineitem = np.genfromtxt("lineitem.tbl",delimiter='|', names=["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"],dtype=['int32','int32','int32','int32','float32','float32','float32','float32','S1','S1','S10','S10','S10','S25','S10','S44'])

def to_integer(dt_time):
  dt_time = dt_time.split("-")
  return 10000*int(dt_time[0]) + 100*int(dt_time[1]) + int(dt_time[2])

vfunc = np.vectorize(to_integer)
l_shipdate = vfunc(lineitem['l_shipdate'])

def q6_computation(shipdate,discount,quantity,extendedprice):
  zeros = tf.zeros_like(discount)
  complete_filter = tf.logical_and(tf.greater_equal(discount,0.05),tf.logical_and(tf.less_equal(discount,0.07),tf.logical_and(tf.less(quantity,24),tf.logical_and(tf.less(shipdate,19950101),tf.greater_equal(shipdate,19940101)))))
  result = tf.reduce_sum(tf.multiply(tf.where(complete_filter,extendedprice,zeros), tf.where(complete_filter,discount,zeros)))
  return result

def q6(): 
  inputs = [tf.convert_to_tensor(l_shipdate, np.int32),tf.convert_to_tensor(lineitem['l_discount'], np.float32),tf.convert_to_tensor(lineitem['l_quantity'], np.float32),tf.convert_to_tensor(lineitem['l_quantity'], np.float32)]
  tpu_computation = tpu.rewrite(q6_computation, inputs)
  tpu_grpc_url = TPUClusterResolver(
    tpu=[os.environ['TPU_NAME']]).get_master()
  with tf.Session(tpu_grpc_url) as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(tpu.initialize_system())
    sess.run(tf.global_variables_initializer())
    res = sess.run(tpu_computation)
    res = sess.run(tpu_computation)
    res = sess.run(tpu_computation)
    res = sess.run(tpu_computation)
    res = sess.run(tpu_computation, options=run_options, run_metadata=run_metadata)
    sess.run(tpu.shutdown_system())
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
    return res


q6()
