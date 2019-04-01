import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
import sys
from subprocess import Popen
import time


l_shipdate = 0
l_discount = 0
l_quantity = 0
l_extendedprice = 0
l_tax = 0
l_returnflag = 0
l_linestatus = 0

l_returnflag_group_size = 0
l_linestatus_group_size = 0


# Turn dates into integers
def date_to_integer(dt_time):
    dt_time = dt_time.split("-")
    return 10000 * int(dt_time[0]) + 100 * int(dt_time[1]) + int(dt_time[2])


def load_input(scale):
    global l_shipdate
    global l_discount
    global l_quantity
    global l_extendedprice
    global l_tax
    global l_returnflag
    global l_linestatus

    os.chdir('/home/pedroholanda/tpch-' + str(scale))
    lineitem = pd.read_csv("lineitem.tbl", sep='|',
                           names=["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
                                  "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
                                  "l_shipdate",
                                  "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
                           dtype={'l_returnflag': 'category', 'l_linestatus': 'category'})
    # Query 01 and 06
    l_shipdate = lineitem["l_shipdate"].values.astype('S10')
    l_discount = lineitem["l_discount"].values.astype('float32')
    l_quantity = lineitem["l_quantity"].values.astype('float32')
    l_extendedprice = lineitem["l_extendedprice"].values.astype('float32')

    # Query 01
    l_tax = lineitem["l_tax"].values.astype('float32')
    l_returnflag = lineitem["l_returnflag"].values.astype('S1')
    l_linestatus = lineitem["l_linestatus"].values.astype('S1')

    # Dictionaries
    l_returnflag[l_returnflag == "A"] = "1"
    l_returnflag[l_returnflag == "N"] = "2"
    l_returnflag[l_returnflag == "R"] = "3"
    l_returnflag = l_returnflag.astype(np.float32, copy=False)

    l_linestatus[l_linestatus == "F"] = "1"
    l_linestatus[l_linestatus == "O"] = "2"
    l_linestatus = l_linestatus.astype(np.float32, copy=False)

    vfunc = np.vectorize(date_to_integer)
    l_shipdate = vfunc(l_shipdate)


def q1_computation(shipdate, returnflag, linestatus, quantity, extendedprice, discount, tax, returnflag_groups_tensors,
                   linestatus_groups_tensors):
    zeros = tf.zeros_like(discount)
    ones = tf.ones_like(discount)
    minus_one = tf.constant(-1.0, dtype=tf.float32)
    returnflag_groups = tf.unstack(returnflag_groups_tensors, l_returnflag_group_size)
    linestatus_groups = tf.unstack(linestatus_groups_tensors, l_linestatus_group_size)
    group_filters = []
    shipdate = tf.less_equal(shipdate, 19980901)
    for returnflag_group in returnflag_groups:
        for linestatus_group in linestatus_groups:
            returnflag_aux = tf.cast(tf.where(tf.equal(returnflag, returnflag_group), ones, zeros), tf.bool)
            linestatus_aux = tf.cast(tf.where(tf.equal(linestatus, linestatus_group), ones, zeros), tf.bool)
            group_filters.append(tf.logical_and(tf.logical_and(returnflag_aux, linestatus_aux), shipdate))
    result = tf.constant([], dtype=tf.float32, shape=[8])
    for group_filter in group_filters:
        sum_qty = tf.reduce_sum(tf.where(group_filter, quantity, zeros))
        sum_base_price = tf.reduce_sum(tf.where(group_filter, extendedprice, zeros))
        sum_disc_price = tf.reduce_sum(
            tf.where(group_filter, tf.multiply(tf.add(ones, tf.multiply(minus_one, discount)), extendedprice), zeros))
        sum_charge = tf.reduce_sum(tf.where(group_filter, tf.multiply(
            tf.multiply(tf.add(ones, tf.multiply(minus_one, discount)), extendedprice), tf.add(ones, tax))
                                            , zeros))
        count = tf.reduce_sum(tf.cast(group_filter, tf.float32))
        avg_qty = tf.div(tf.reduce_sum(tf.where(group_filter, quantity, zeros)), tf.reduce_sum(count))
        avg_price = tf.div(tf.reduce_sum(tf.where(group_filter, extendedprice, zeros)), tf.reduce_sum(count))
        avg_disc = tf.div(tf.reduce_sum(tf.where(group_filter, discount, zeros)), tf.reduce_sum(count))
        result = tf.concat([result, tf.stack(
            [sum_qty, sum_base_price, sum_disc_price, sum_charge, avg_qty, avg_price, avg_disc, count])], axis=0)
    result = tf.reshape(result, [l_returnflag_group_size * l_linestatus_group_size + 1, 8])
    return result


def q1():
    global l_returnflag_group_size
    global l_linestatus_group_size
    returnflag_groups = np.unique(l_returnflag)
    linestatus_groups = np.unique(l_linestatus)
    l_returnflag_group_size = len(returnflag_groups)
    l_linestatus_group_size = len(linestatus_groups)
    inputs = [tf.convert_to_tensor(l_shipdate, np.float32), tf.convert_to_tensor(l_returnflag, np.float32),
              tf.convert_to_tensor(l_linestatus, np.float32), tf.convert_to_tensor(l_quantity, np.float32),
              tf.convert_to_tensor(l_extendedprice, np.float32), tf.convert_to_tensor(l_discount, np.float32),
              tf.convert_to_tensor(l_tax, np.float32), tf.convert_to_tensor(returnflag_groups, np.float32),
              tf.convert_to_tensor(linestatus_groups, np.float32)]
    tpu_computation = tpu.rewrite(q1_computation, inputs)
    tpu_grpc_url = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']]).get_master()
    with tf.Session(tpu_grpc_url) as sess:
        sess.run(tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        for i in range(0,5):
            res = sess.run(tpu_computation)
        sess.run(tpu.shutdown_system())
        print (res)
        return res


def q6_computation(shipdate, discount, quantity, extendedprice):
    zeros = tf.zeros_like(discount)
    complete_filter = tf.logical_and(tf.greater_equal(discount, 0.05), tf.logical_and(tf.less_equal(discount, 0.07),
                                                                                      tf.logical_and(
                                                                                          tf.less(quantity, 24),
                                                                                          tf.logical_and(
                                                                                              tf.less(shipdate,
                                                                                                      19950101),
                                                                                              tf.greater_equal(shipdate,
                                                                                                               19940101)))))
    result = tf.reduce_sum(
        tf.multiply(tf.where(complete_filter, extendedprice, zeros), tf.where(complete_filter, discount, zeros)))
    return result


def q6():
    inputs = [tf.convert_to_tensor(l_shipdate, np.float32), tf.convert_to_tensor(l_discount, np.float32),
              tf.convert_to_tensor(l_quantity, np.float32), tf.convert_to_tensor(l_extendedprice, np.float32)]
    tpu_computation = tpu.rewrite(q6_computation, inputs)
    tpu_grpc_url = TPUClusterResolver(
        tpu=[os.environ['TPU_NAME']]).get_master()
    with tf.Session(tpu_grpc_url) as sess:
        sess.run(tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        for i in range(0,5):
            res = sess.run(tpu_computation)
        sess.run(tpu.shutdown_system())
        print(res)
        return res


def run_tpch(scale):
    load_input(scale)
    Popen('capture_tpu_profile --tpu=pedroholanda --logdir=${STORAGE_BUCKET}/res --num_tracing_attempts=10', shell=True,stdin=None, stdout=None, stderr=None, close_fds=True)
    time.sleep(5)
    q1()
    # q6()


if __name__ == "__main__":
    scale = int(sys.argv[1])
    run_tpch(scale)
