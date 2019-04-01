import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
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


# This is not implemented on TPUs, which means returning a list of elements would have to be performed on CPUs
# Boolean Mask also not supported
# filter = tf.where(tf.logical_and(tf.greater_equal(quantity,10),tf.less_equal(quantity,24)))
def filter_sum_computation(quantity):
    zeros = tf.zeros_like(quantity)
    result = tf.reduce_sum(
        tf.where(tf.logical_and(tf.greater_equal(quantity, 10), tf.less_equal(quantity, 24)), quantity, zeros))
    return result


def filter_sum():
    inputs = [tf.convert_to_tensor(l_quantity, np.float32)]
    tpu_computation = tpu.rewrite(filter_sum_computation, inputs)
    tpu_grpc_url = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']]).get_master()
    with tf.Session(tpu_grpc_url) as sess:
        sess.run(tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        for i in range (0,5):
            res = sess.run(tpu_computation)
        sess.run(tpu.shutdown_system())
        print(res)
        return res


def filter_computation(quantity):
    result = tf.logical_and(tf.greater_equal(quantity, 10), tf.less_equal(quantity, 24))
    return result


def filter():
    inputs = [tf.convert_to_tensor(l_quantity, np.float32)]
    tpu_computation = tpu.rewrite(filter_computation, inputs)
    tpu_grpc_url = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']]).get_master()
    with tf.Session(tpu_grpc_url) as sess:
        sess.run(tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        for i in range (0,5):
            res = sess.run(tpu_computation)
        sess.run(tpu.shutdown_system())
        print(l_quantity[res])
        return res


def group_by_computation(quantity, returnflag, unique_groups):
    ones = tf.ones_like(returnflag)
    zeros = tf.zeros_like(returnflag)
    returnflag_groups = tf.unstack(unique_groups, 3)
    result = []
    for returnflag_group in returnflag_groups:
        returnflag_filter = tf.cast(tf.where(tf.equal(returnflag, returnflag_group), ones, zeros), tf.bool)
        sum_qty = tf.reduce_sum(tf.where(returnflag_filter, quantity, zeros))
        result.extend([returnflag_group, sum_qty])
    return result


def group_by():
    unique_groups = np.unique(l_returnflag)
    inputs = [tf.convert_to_tensor(l_quantity, np.float32), tf.convert_to_tensor(l_returnflag, np.float32),
              tf.convert_to_tensor(unique_groups, np.float32)]
    tpu_computation = tpu.rewrite(group_by_computation, inputs)
    tpu_grpc_url = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']]).get_master()
    with tf.Session(tpu_grpc_url) as sess:
        sess.run(tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        for i in range (0,5):
            res = sess.run(tpu_computation)
        sess.run(tpu.shutdown_system())
        print(res)


def aggregation_computation(quantity):
    result = tf.reduce_sum(quantity)
    return result


def aggregation():
    inputs = [tf.convert_to_tensor(l_quantity, np.float32)]
    tpu_computation = tpu.rewrite(aggregation_computation, inputs)
    tpu_grpc_url = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']]).get_master()
    with tf.Session(tpu_grpc_url) as sess:
        sess.run(tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        for i in range (0,5):
            res = sess.run(tpu_computation)
        sess.run(tpu.shutdown_system())
        print(res)
        return res


def order_by_limit_computation(quantity):
    result, indices = tf.nn.top_k(quantity, 10, True)
    return result


def order_by_limit():
    inputs = [tf.convert_to_tensor(l_quantity, np.float32)]
    tpu_computation = tpu.rewrite(order_by_limit_computation, inputs)
    tpu_grpc_url = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']]).get_master()
    with tf.Session(tpu_grpc_url) as sess:
        sess.run(tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        for i in range (0,5):
            res = sess.run(tpu_computation)
        sess.run(tpu.shutdown_system())
        print(res)
        return res


def join_computation(lineitem_orderkey, order_orderkey):
    orders = tf.unstack(order_orderkey, 150000)
    zeros = tf.zeros_like(lineitem_orderkey)
    for order in orders:
        aux = tf.reduce_sum(tf.where(tf.equal(lineitem_orderkey, order), lineitem_orderkey, zeros))
    result = aux
    return result


def join():
    inputs = [tf.convert_to_tensor(l_orderkey, np.int32), tf.convert_to_tensor(o_orderkey, np.int32)]
    tpu_computation = tpu.rewrite(join_computation, inputs)
    tpu_grpc_url = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']]).get_master()
    with tf.Session(tpu_grpc_url) as sess:
        sess.run(tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        for i in range (0,5):
            res = sess.run(tpu_computation)
        sess.run(tpu.shutdown_system())
        print(res)
        return res


def run_micro(scale):
    load_input(scale)
    filter_sum()
    filter()
    aggregation()
    order_by_limit()
    group_by()
    # join()


if __name__ == "__main__":
    scale = int(sys.argv[1])
    run_micro(scale)
