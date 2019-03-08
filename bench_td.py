import numpy as np
import timeit, csv
import os
import tensorflow as tf
from tensorflow.python.client import timeline


os.chdir('tpctpu')

lineitem = np.genfromtxt("lineitem.tbl",delimiter='|', names=["l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"],dtype=['int32','int32','int32','int32','float32','float32','float32','float32','S1','S1','S10','S10','S10','S25','S10','S44'])

def to_integer(dt_time):
  dt_time = dt_time.split("-")
  return 10000*int(dt_time[0]) + 100*int(dt_time[1]) + int(dt_time[2])

vfunc = np.vectorize(to_integer)
l_shipdate = vfunc(lineitem['l_shipdate'])


def q1():
  l_returnflag = tf.placeholder(dtype=tf.string, shape=(None,))
  l_linestatus = tf.placeholder(dtype=tf.string, shape=(None,))
  y, idx = unique(l_returnflag)


def q6(): 
  shipdate = tf.placeholder(dtype=tf.int32, shape=(None,))
  discount = tf.placeholder(dtype=tf.float32, shape=(None,))
  quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
  extendedprice = tf.placeholder(dtype=tf.float32, shape=(None,))
  zeros = tf.zeros_like(discount)
  complete_filter = tf.logical_and(tf.greater_equal(discount,0.05),tf.logical_and(tf.less_equal(discount,0.07),tf.logical_and(tf.less(quantity,24),tf.logical_and(tf.less(shipdate,19950101),tf.greater_equal(shipdate,19940101)))))
  result = tf.reduce_sum(tf.multiply(tf.where(complete_filter,extendedprice,zeros), tf.where(complete_filter,discount,zeros)))
  with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    res = sess.run(result, feed_dict={
        shipdate: l_shipdate,
        discount: lineitem['l_discount'],
        quantity: lineitem['l_quantity'],
        extendedprice: lineitem['l_extendedprice']
      })
    res = sess.run(result, feed_dict={
      shipdate: l_shipdate,
      discount: lineitem['l_discount'],
      quantity: lineitem['l_quantity'],
      extendedprice: lineitem['l_extendedprice']
    })
    res = sess.run(result, feed_dict={
      shipdate: l_shipdate,
      discount: lineitem['l_discount'],
      quantity: lineitem['l_quantity'],
      extendedprice: lineitem['l_extendedprice']
    })
    res = sess.run(result, feed_dict={
      shipdate: l_shipdate,
      discount: lineitem['l_discount'],
      quantity: lineitem['l_quantity'],
      extendedprice: lineitem['l_extendedprice']
    })
    res = sess.run(result, feed_dict={
        shipdate: l_shipdate,
        discount: lineitem['l_discount'],
        quantity: lineitem['l_quantity'],
        extendedprice: lineitem['l_extendedprice']
      }, options=run_options, run_metadata=run_metadata)
     # Create the Timeline object, and write it to a json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
      # print res
  return res


q6()
