import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.python.client import timeline

os.chdir('TPC-H/tpch-1/')
#Pandas IO faster than Numpy
lineitem_df = pd.read_csv("lineitem.tbl", sep='|',
                          names=["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
                                 "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate",
                                 "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
                          dtype={'l_returnflag': 'category', 'l_linestatus': 'category'})
#Query 01 and 06
l_shipdate = lineitem_df["l_shipdate"].values.astype('S10')
l_discount = lineitem_df["l_discount"].values.astype('float32')
l_quantity = lineitem_df["l_quantity"].values.astype('float32')
l_extendedprice = lineitem_df["l_extendedprice"].values.astype('float32')

#Query 01
l_tax = lineitem_df["l_tax"].values.astype('float32')
l_returnflag = lineitem_df["l_returnflag"].values.astype('S1')
l_linestatus = lineitem_df["l_linestatus"].values.astype('S1')
del lineitem_df


# Groupings:
# tf.where
# Tf.gather

# Map
# Scatter for building
# Gather for scanning

def date_to_integer(dt_time):
    dt_time = dt_time.split("-")
    return 10000 * int(dt_time[0]) + 100 * int(dt_time[1]) + int(dt_time[2])

vfunc = np.vectorize(date_to_integer)
l_shipdate = vfunc(l_shipdate)

l_returnflag[l_returnflag=="R"] = "1"
l_returnflag[l_returnflag=="N"] = "2"
l_returnflag[l_returnflag=="A"] = "3"
l_returnflag = l_returnflag.astype(np.float32, copy=False)

l_linestatus[l_linestatus=="O"] = "1"
l_linestatus[l_linestatus=="F"] = "2"
l_linestatus = l_linestatus.astype(np.float32, copy=False)

#Need to use while functions from TF to transform tensor into lists
#Because Tensors are not iterable
i = tf.constant(0)
while_condition = lambda i: tf.less(i, input_placeholder[1, 1])

def body_get_const(i):
    # do something here which you want to do in your loop
    # increment i
    return i



# Might be possible to not hardcode the groups if I use tf.unique y, idx = tf.unique(l_returnflag)
# Sorting Might be possible with tf.nn.top_k and tf.gather_nd with tf.meshgrid
def q1():
    shipdate = tf.placeholder(dtype=tf.int32, shape=(None,))
    returnflag = tf.placeholder(dtype=tf.float32, shape=(None,))
    linestatus = tf.placeholder(dtype=tf.float32, shape=(None,))
    quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
    extendedprice = tf.placeholder(dtype=tf.float32, shape=(None,))
    discount = tf.placeholder(dtype=tf.float32, shape=(None,))
    tax = tf.placeholder(dtype=tf.float32, shape=(None,))
    zeros = tf.zeros_like(discount)
    ones = tf.ones_like(discount)
    minus_one = tf.constant(-1.0,dtype=tf.float32)

    #Performing the groups
    returnflag_groups_tensors,idx = tf.unique(returnflag)
    linestatus_groups_tensors,idx = tf.unique(linestatus)
    aux = tf.zeros_like(returnflag_groups_tensors)
    returnflag_groups = tf.stack([aux,returnflag_groups_tensors], axis=1)
    aux = tf.zeros_like(linestatus_groups_tensors)
    linestatus_groups = tf.stack([aux,linestatus_groups_tensors], axis=1)
    returnflag_groups = tf.unstack(returnflag_groups_tensors,3) # This needs to be hardcoded :-( this can come from HyperLogLog or Index
    linestatus_groups = tf.unstack(linestatus_groups_tensors,2) # This needs to be hardcoded :-( this can come from HyperLogLog or Index
    group_filters = []
    shipdate = tf.less_equal(shipdate, 19980901)
    for returnflag_group in returnflag_groups:
        for linestatus_group in linestatus_groups:
            returnflag_aux = tf.cast(tf.where(tf.equal(returnflag,returnflag_group),ones,zeros),tf.bool)
            linestatus_aux = tf.cast(tf.where(tf.equal(linestatus,linestatus_group),ones,zeros),tf.bool)
            group_filters.append(tf.logical_and(tf.logical_and(returnflag_aux,linestatus_aux),shipdate))

    result = []
    for group_filter in group_filters:
        sum_qty = tf.reduce_sum(tf.where(group_filter,quantity,zeros))
        sum_base_price = tf.reduce_sum(tf.where(group_filter,extendedprice,zeros))
        sum_disc_price = tf.reduce_sum(tf.where(group_filter,tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),zeros))
        sum_charge = tf.reduce_sum(tf.where(group_filter, tf.multiply(tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),tf.add(ones,tax))
    ,zeros))
        count = tf.reduce_sum(tf.cast(group_filter,tf.float32))
        avg_qty = tf.div(tf.reduce_sum(tf.where(group_filter,quantity,zeros)),tf.reduce_sum(count)) #int32 is slower  
        avg_price = tf.div(tf.reduce_sum(tf.where(group_filter,extendedprice,zeros)),tf.reduce_sum(count)) #int32 is slower  
        avg_disc = tf.div(tf.reduce_sum(tf.where(group_filter,discount,zeros)),tf.reduce_sum(count)) #int32 is slower  
        result_aux = sum_qty,sum_base_price,sum_disc_price,sum_charge,avg_qty,avg_price,avg_disc,count
        result.extend(result_aux)
    
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(result, feed_dict={
            shipdate: l_shipdate,
            returnflag: l_returnflag,
            linestatus: l_linestatus,
            quantity: l_quantity,
            extendedprice: l_extendedprice,
            discount: l_discount,
            tax: l_tax
        })
    print res

def q1_hard_coded_groups():
    shipdate = tf.placeholder(dtype=tf.int32, shape=(None,))
    returnflag = tf.placeholder(dtype=tf.float32, shape=(None,))
    linestatus = tf.placeholder(dtype=tf.float32, shape=(None,))
    quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
    extendedprice = tf.placeholder(dtype=tf.float32, shape=(None,))
    discount = tf.placeholder(dtype=tf.float32, shape=(None,))
    tax = tf.placeholder(dtype=tf.float32, shape=(None,))
    zeros = tf.zeros_like(discount)
    ones = tf.ones_like(discount)
    minus_one = tf.constant(-1.0,dtype=tf.float32)
    two = tf.constant(2.0,dtype=tf.float32)
    three = tf.constant(3.0,dtype=tf.float32)

    R = tf.cast(tf.where(tf.equal(returnflag,ones),ones,zeros),tf.bool)
    N = tf.cast(tf.where(tf.equal(returnflag,two),ones,zeros),tf.bool)
    A = tf.cast(tf.where(tf.equal(returnflag,three),ones,zeros),tf.bool)

    O = tf.cast(tf.where(tf.equal(linestatus,ones),ones,zeros),tf.bool)
    F = tf.cast(tf.where(tf.equal(linestatus,two),ones,zeros),tf.bool)
    shipdate = tf.less_equal(shipdate, 19980901)

    # AF
    group_filter_1 = tf.logical_and(tf.logical_and(A,F),shipdate)
    # NF
    group_filter_2 = tf.logical_and(tf.logical_and(N,F),shipdate)
    # NO
    group_filter_3 = tf.logical_and(tf.logical_and(N,O),shipdate)
    # RF
    group_filter_4 = tf.logical_and(tf.logical_and(R,F),shipdate)
    group_filters = group_filter_1,group_filter_2,group_filter_3,group_filter_4

    result = []
    for group_filter in group_filters:
        sum_qty = tf.reduce_sum(tf.where(group_filter,quantity,zeros))
        sum_base_price = tf.reduce_sum(tf.where(group_filter,extendedprice,zeros))
        sum_disc_price = tf.reduce_sum(tf.where(group_filter,tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),zeros))
        sum_charge = tf.reduce_sum(tf.where(group_filter, tf.multiply(tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),tf.add(ones,tax))
    ,zeros))
        count = tf.reduce_sum(tf.cast(group_filter,tf.float32))
        avg_qty = tf.div(tf.reduce_sum(tf.where(group_filter,quantity,zeros)),tf.reduce_sum(count)) #int32 is slower  
        avg_price = tf.div(tf.reduce_sum(tf.where(group_filter,extendedprice,zeros)),tf.reduce_sum(count)) #int32 is slower  
        avg_disc = tf.div(tf.reduce_sum(tf.where(group_filter,discount,zeros)),tf.reduce_sum(count)) #int32 is slower  
        result_aux = sum_qty,sum_base_price,sum_disc_price,sum_charge,avg_qty,avg_price,avg_disc,count
        result.extend(result_aux)
    
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        res = sess.run(result, feed_dict={
            shipdate: l_shipdate,
            returnflag: l_returnflag,
            linestatus: l_linestatus,
            quantity: l_quantity,
            extendedprice: l_extendedprice,
            discount: l_discount,
            tax: l_tax
        })
        print res


def q6():
    with tf.device('/device:GPU:0'):
        shipdate = tf.placeholder(dtype=tf.int32, shape=(None,))
        discount = tf.placeholder(dtype=tf.float32, shape=(None,))
        quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
        extendedprice = tf.placeholder(dtype=tf.float32, shape=(None,))
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
        with tf.Session() as sess:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            res = sess.run(result, feed_dict={
                shipdate: l_shipdate,
                discount: l_discount,
                quantity: l_quantity,
                extendedprice: l_extendedprice
            })
            res = sess.run(result, feed_dict={
                shipdate: l_shipdate,
                discount: l_discount,
                quantity: l_quantity,
                extendedprice: l_extendedprice
            })
            res = sess.run(result, feed_dict={
                shipdate: l_shipdate,
                discount: l_discount,
                quantity: l_quantity,
                extendedprice: l_extendedprice
            })
            res = sess.run(result, feed_dict={
                shipdate: l_shipdate,
                discount: l_discount,
                quantity: l_quantity,
                extendedprice: l_extendedprice
            })
            res = sess.run(result, feed_dict={
                shipdate: l_shipdate,
                discount: l_discount,
                quantity: l_quantity,
                extendedprice: l_extendedprice
            }, options=run_options, run_metadata=run_metadata)
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)
            print res
        return res

q1()
# q6()
