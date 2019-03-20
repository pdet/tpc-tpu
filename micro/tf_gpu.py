import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.python.client import timeline
import sys

o_orderkey = 0
l_orderkey = 0
l_quantity = 0
l_returnflag = 0


def load_input(scale):
    global o_orderkey
    global l_orderkey
    global l_quantity
    global l_returnflag
    os.chdir('/home/pedroholanda/tpch-' + str(scale))
    lineitem = pd.read_csv("lineitem.tbl", sep='|',
                           names=["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
                                  "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
                                  "l_shipdate",
                                  "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
                           dtype={'l_returnflag': 'category', 'l_linestatus': 'category'})
    orders = pd.read_csv("orders.tbl", sep='|',
                         names=["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
                                "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
                         dtype={'o_orderstatus': 'category', 'o_orderpriority': 'category'})

    o_orderkey = orders["o_orderkey"].values.astype('int32')
    l_orderkey = lineitem["l_orderkey"].values.astype('int32')
    l_quantity = lineitem["l_quantity"].values.astype('float32')
    l_returnflag = lineitem["l_returnflag"].values.astype('S1')
    l_returnflag[l_returnflag == "A"] = "1"
    l_returnflag[l_returnflag == "N"] = "2"
    l_returnflag[l_returnflag == "R"] = "3"
    l_returnflag = l_returnflag.astype(np.float32, copy=False)
    os.chdir('/home/pedroholanda/result/')


def filter_sum(scale):
    with tf.device('/device:GPU:0'):
        quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
        zeros = tf.zeros_like(quantity)
        result = tf.reduce_sum(
            tf.where(tf.logical_and(tf.greater_equal(quantity, 10), tf.less_equal(quantity, 24)), quantity, zeros))
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        }, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_filter_sum_gpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def filter(scale):
    with tf.device('/device:GPU:0'):
        quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
        filter = tf.where(tf.logical_and(tf.greater_equal(quantity, 10), tf.less_equal(quantity, 24)))
        result = tf.gather(quantity, filter)
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        }, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_filter_gpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def aggregation(scale):
    with tf.device('/device:GPU:0'):
        quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
        result = tf.reduce_sum(quantity)
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        }, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_aggregation_gpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def group_by(scale):
    quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
    returnflag = tf.placeholder(dtype=tf.float32, shape=(None,))
    ones = tf.ones_like(quantity)
    zeros = tf.zeros_like(quantity)
    returnflag_groups_tensors, idx = tf.unique(returnflag)
    with tf.device('/device:GPU:0'):
        returnflag_groups = tf.unstack(returnflag_groups_tensors, 3)
        result = tf.constant(0.0, dtype=tf.float32, shape=[2])
        for returnflag_group in returnflag_groups:
            returnflag_filter = tf.cast(tf.where(tf.equal(returnflag, returnflag_group), ones, zeros), tf.bool)
            sum_qty = tf.reduce_sum(tf.where(returnflag_filter, quantity, zeros))
            result = tf.concat([result, tf.stack([returnflag_group, sum_qty])], axis=0)
        result = tf.reshape(result, [4, 2])[1:]
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(result, feed_dict={
            quantity: l_quantity,
            returnflag: l_returnflag
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity,
            returnflag: l_returnflag
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity,
            returnflag: l_returnflag
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity,
            returnflag: l_returnflag
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity,
            returnflag: l_returnflag
        }, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_grouping_gpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def order_by_limit(scale, quantity_size):
    with tf.device('/device:GPU:0'):
        quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
        sorted_a, indices = tf.nn.top_k(quantity, quantity_size, True)
        result = sorted_a
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        })
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        }, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_order_gpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def join(scale, order_size):
    with tf.device('/device:GPU:0'):
        lineitem_orderkey = tf.placeholder(dtype=tf.int32, shape=(None,))
        zeros = tf.zeros_like(lineitem_orderkey)
        order_orderkey = tf.placeholder(dtype=tf.int32, shape=(None,))
        orders = tf.unstack(order_orderkey, order_size)
        # join_idx = tf.constant(0.0,dtype=tf.float32)
        for order in orders:
            # join_idx = tf.stack(join_idx,(tf.where(tf.equal(lineitem_orderkey,order))))
            aux = tf.reduce_sum(tf.where(tf.equal(lineitem_orderkey, order), lineitem_orderkey, zeros))
        result = aux
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(result, feed_dict={
            lineitem_orderkey: l_orderkey,
            order_orderkey: o_orderkey
        })
        res = sess.run(result, feed_dict={
            lineitem_orderkey: l_orderkey,
            order_orderkey: o_orderkey
        })
        res = sess.run(result, feed_dict={
            lineitem_orderkey: l_orderkey,
            order_orderkey: o_orderkey
        })
        res = sess.run(result, feed_dict={
            lineitem_orderkey: l_orderkey,
            order_orderkey: o_orderkey
        })
        res = sess.run(result, feed_dict={
            lineitem_orderkey: l_orderkey,
            order_orderkey: o_orderkey
        }, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_join_gpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def run_micro(scale):
    load_input(scale)
    filter_sum(scale)
    filter(scale)
    aggregation(scale)
    group_by(scale)
    order_by_limit(scale, 10)
    # join(scale,150000)
    return 0


if __name__ == "__main__":
    scale = int(sys.argv[1])
    run_micro(scale)
