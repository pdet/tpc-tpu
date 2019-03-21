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
    # os.chdir('/home/pedroholanda/tpch-' + str(scale))
    os.chdir('/Users/holanda/Documents/Projects/tpc-tpu/tpch-dbgen')

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
    # os.chdir('/home/pedroholanda/result/')
    os.chdir('/Users/holanda/Documents/Projects/tpc-tpu/Results')



def filter_sum(scale):
    quantity = tf.placeholder(dtype=tf.float32, shape=(None,), name='quantity')
    zeros = tf.zeros_like(quantity, name='zero')
    result = tf.reduce_sum(
        tf.where(tf.logical_and(tf.greater_equal(quantity, 10, name='GEQ'), tf.less_equal(quantity, 24, name='LEQ'), name='AND'), quantity, zeros, name='FILTER'), name='SUM')
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range (0,4):
            res = sess.run(result, feed_dict={
                quantity: l_quantity
            })
        writer = tf.summary.FileWriter("./filter_sum/", sess.graph)
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        }, options=run_options, run_metadata=run_metadata)
        writer.close()
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_filter_sum_cpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def filter(scale):
    quantity = tf.placeholder(dtype=tf.float32, shape=(None,), name='quantity')
    filter = tf.where(tf.logical_and(tf.greater_equal(quantity, 10, name='GEQ'), tf.less_equal(quantity, 24, name='LEQ'), name='AND'), name='FILTER')
    result = tf.gather(quantity, filter, name='GATHER')
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range (0,4):
            res = sess.run(result, feed_dict={
                quantity: l_quantity
            })
        writer = tf.summary.FileWriter("./filter/", sess.graph)
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        }, options=run_options, run_metadata=run_metadata)
        writer.close()
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_filter_cpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def aggregation(scale):
    quantity = tf.placeholder(dtype=tf.float32, shape=(None,), name='quantity')
    result = tf.reduce_sum(quantity, name='SUM')
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range (0,4):
            res = sess.run(result, feed_dict={
                quantity: l_quantity
            })
        writer = tf.summary.FileWriter("./aggregation/", sess.graph)
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        }, options=run_options, run_metadata=run_metadata)
        writer.close()
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_aggregation_cpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def group_by(scale):
    quantity = tf.placeholder(dtype=tf.float32, shape=(None,), name='quantity')
    returnflag = tf.placeholder(dtype=tf.float32, shape=(None,), name='returnflag')
    ones = tf.ones_like(quantity, name='one')
    zeros = tf.zeros_like(quantity, name='zero')
    returnflag_groups_tensors, idx = tf.unique(returnflag, name='UNIQUE')
    returnflag_groups = tf.unstack(returnflag_groups_tensors, 3, name='UNSTACK')
    result = tf.constant(0.0, dtype=tf.float32, shape=[2], name='result')
    for returnflag_group in returnflag_groups:
        returnflag_filter = tf.cast(tf.where(tf.equal(returnflag, returnflag_group, name='EQUAL'), ones, zeros, name='FILTER'), tf.bool, name='CAST')
        sum_qty = tf.reduce_sum(tf.where(returnflag_filter, quantity, zeros, name='FILTER'), name='SUM')
        result = tf.concat([result, tf.stack([returnflag_group, sum_qty])], axis=0, name='CONCAT')
    result = tf.reshape(result, [4, 2], name='RESHAPE')[1:]
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range (0,4):
            res = sess.run(result, feed_dict={
                quantity: l_quantity,
                returnflag: l_returnflag
            })
        writer = tf.summary.FileWriter("./group_by/", sess.graph)
        res = sess.run(result, feed_dict={
            quantity: l_quantity,
            returnflag: l_returnflag
        }, options=run_options, run_metadata=run_metadata)
        writer.close()
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_grouping_cpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def order_by_limit(scale, quantity_size):
    quantity = tf.placeholder(dtype=tf.float32, shape=(None,), name='quantity')
    result = tf.nn.top_k(quantity, quantity_size, True, name='TOP_K')
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range (0,4):
            res = sess.run(result, feed_dict={
                quantity: l_quantity
            })
        writer = tf.summary.FileWriter("./order_by_limit/", sess.graph)
        res = sess.run(result, feed_dict={
            quantity: l_quantity
        }, options=run_options, run_metadata=run_metadata)
        writer.close()
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_order_cpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res



def join(scale, order_size):

    lineitem_orderkey = tf.placeholder(dtype=tf.int32, shape=(None,))
    zeros = tf.zeros_like(lineitem_orderkey)
    order_orderkey = tf.placeholder(dtype=tf.int32, shape=(None,))
    orders = tf.unstack(order_orderkey, order_size)
    # Track both the loop index and summation in a tuple in the form (index, summation)
    index_summation = (tf.constant(1), tf.constant(0.0))

    # The loop condition, note the loop condition is 'i < n-1'
    def condition(index, summation):
        return tf.less(index, tf.subtract(tf.shape(orders)[0], 1))

    # The loop body, this will return a result tuple in the same form (index, summation)
    def body(index, summation):
        # x_i = tf.gather(columns, index)
        # x_ip1 = tf.gather(columns, tf.add(index, 1))

        # first_term = tf.square(tf.subtract(x_ip1, tf.square(x_i)))
        # second_term = tf.square(tf.subtract(x_i, 1.0))
        summand = tf.reduce_sum(tf.where(tf.equal(lineitem_orderkey, order), lineitem_orderkey, zeros))

        return tf.add(index, 1), tf.add(summation, summand)
    result =  tf.while_loop(condition, body, index_summation,parallel_iterations=order_size,maximum_iterations=order_size)[1]
    # join_idx = tf.constant(0.0,dtype=tf.float32)
    # for order in orders:
    #     # join_idx = tf.stack(join_idx,(tf.where(tf.equal(lineitem_orderkey,order))))
    #     aux = tf.reduce_sum(tf.where(tf.equal(lineitem_orderkey, order), lineitem_orderkey, zeros))
    # result = aux
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range (0,4):
            res = sess.run(result, feed_dict={
                lineitem_orderkey: l_orderkey,
                order_orderkey: o_orderkey
            })
        writer = tf.summary.FileWriter("./join/", sess.graph)
        res = sess.run(result, feed_dict={
            lineitem_orderkey: l_orderkey,
            order_orderkey: o_orderkey
        }, options=run_options, run_metadata=run_metadata)
        writer.close()
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_join_cpu_' + str(scale) + '.json', 'w') as f:
            f.write(ctf)
        print res
    return res


def run_micro(scale):
    load_input(scale)
    # filter_sum(scale)
    # filter(scale)
    # aggregation(scale)
    # group_by(scale)
    # order_by_limit(scale, 10)
    join(scale,150000)
    return 0


if __name__ == "__main__":
    scale = int(sys.argv[1])
    run_micro(scale)
