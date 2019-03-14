import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.python.client import timeline

# Type string is not supported on TPU devices.
# Type float64 is not supported on TPU devices.

os.chdir('tpch-dbgen')

# Pandas IO faster than Numpy
lineitem_df = pd.read_csv("lineitem.tbl", sep='|',
                          names=["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
                                 "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate",
                                 "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
                          dtype={'l_returnflag': 'category', 'l_linestatus': 'category'})
# Query 01 and 06
l_shipdate = lineitem_df["l_shipdate"].values.astype('S10')
l_discount = lineitem_df["l_discount"].values.astype('float32')
l_quantity = lineitem_df["l_quantity"].values.astype('float32')
l_extendedprice = lineitem_df["l_extendedprice"].values.astype('float32')

# Query 01
l_tax = lineitem_df["l_tax"].values.astype('float32')
l_returnflag = lineitem_df["l_returnflag"].values.astype('S1')
l_linestatus = lineitem_df["l_linestatus"].values.astype('S1')
del lineitem_df


def to_integer(dt_time):
    dt_time = dt_time.split("-")
    return 10000 * int(dt_time[0]) + 100 * int(dt_time[1]) + int(dt_time[2])


vfunc = np.vectorize(to_integer)
l_shipdate = vfunc(l_shipdate)

l_returnflag[l_returnflag == "R"] = "1"
l_returnflag[l_returnflag == "N"] = "2"
l_returnflag[l_returnflag == "A"] = "3"
l_returnflag = l_returnflag.astype(np.float32, copy=False)

l_linestatus[l_linestatus == "O"] = "1"
l_linestatus[l_linestatus == "F"] = "2"
l_linestatus = l_linestatus.astype(np.float32, copy=False)


def q1_computation(shipdate, returnflag, linestatus, quantity, extendedprice, discount, tax):
    zeros = tf.zeros_like(discount)
    ones = tf.ones_like(discount)
    minus_one = tf.constant(-1.0, dtype=tf.float32)
    #Performing the groups
    returnflag_groups_tensors,idx = tf.unique(returnflag)
    linestatus_groups_tensors,idx = tf.unique(linestatus)
    aux = tf.zeros_like(returnflag_groups_tensors)
    returnflag_groups = tf.stack([aux,returnflag_groups_tensors], axis=1)
    aux = tf.zeros_like(linestatus_groups_tensors)
    linestatus_groups = tf.stack([aux,linestatus_groups_tensors], axis=1)
    returnflag_groups = tf.unstack(returnflag_groups_tensors,3) # This needs to be hardcoded :-(
    linestatus_groups = tf.unstack(linestatus_groups_tensors,2) # This needs to be hardcoded :-(
    group_filters = []
    shipdate = tf.less_equal(shipdate, 19980901)
    for returnflag_group in returnflag_groups:
        for linestatus_group in linestatus_groups:
            returnflag_aux = tf.cast(tf.where(tf.equal(returnflag,returnflag_group),ones,zeros),tf.bool)
            linestatus_aux = tf.cast(tf.where(tf.equal(linestatus,linestatus_group),ones,zeros),tf.bool)
            group_filters.append(tf.logical_and(tf.logical_and(returnflag_aux,linestatus_aux),shipdate))
    result = []
    for group_filter in group_filters:
        sum_qty = tf.reduce_sum(tf.where(group_filter, quantity, zeros))
        sum_base_price = tf.reduce_sum(tf.where(group_filter, extendedprice, zeros))
        sum_disc_price = tf.reduce_sum(
            tf.where(group_filter, tf.multiply(tf.add(ones, tf.multiply(minus_one, discount)), extendedprice), zeros))
        sum_charge = tf.reduce_sum(tf.where(group_filter, tf.multiply(
            tf.multiply(tf.add(ones, tf.multiply(minus_one, discount)), extendedprice), tf.add(ones, tax))
                                            , zeros))
        count = tf.reduce_sum(tf.cast(group_filter, tf.float32))
        avg_qty = tf.div(tf.reduce_sum(tf.where(group_filter, quantity, zeros)),
                         tf.reduce_sum(count))  # int32 is slower
        avg_price = tf.div(tf.reduce_sum(tf.where(group_filter, extendedprice, zeros)),
                           tf.reduce_sum(count))  # int32 is slower
        avg_disc = tf.div(tf.reduce_sum(tf.where(group_filter, discount, zeros)),
                          tf.reduce_sum(count))  # int32 is slower
        result_aux = sum_qty, sum_base_price, sum_disc_price, sum_charge, avg_qty, avg_price, avg_disc, count
        result.extend(result_aux)


def q1():
    inputs = [tf.convert_to_tensor(l_shipdate, np.int32), tf.convert_to_tensor(l_returnflag, np.float32),
              tf.convert_to_tensor(l_linestatus, np.float32), tf.convert_to_tensor(l_quantity, np.float32),
              tf.convert_to_tensor(l_extendedprice, np.float32), tf.convert_to_tensor(l_discount, np.float32),
              tf.convert_to_tensor(l_tax, np.float32)]
    tpu_computation = tpu.rewrite(q1_computation, inputs)
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
    inputs = [tf.convert_to_tensor(l_shipdate, np.int32), tf.convert_to_tensor(l_discount, np.float32),
              tf.convert_to_tensor(l_quantity, np.float32), tf.convert_to_tensor(l_extendedprice, np.float32)]
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
        print(res)
        return res


q1()
# q6()
