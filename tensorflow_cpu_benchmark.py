import numpy as np
import pandas as pd
import timeit, csv
import os
import tensorflow as tf
from tensorflow.python.client import timeline

os.chdir('tpch-dbgen')
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

def to_integer(dt_time):
    dt_time = dt_time.split("-")
    return 10000 * int(dt_time[0]) + 100 * int(dt_time[1]) + int(dt_time[2])


vfunc = np.vectorize(to_integer)
l_shipdate = vfunc(l_shipdate)

# Might be possible to not hardcode the groups if I use tf.unique y, idx = tf.unique(l_returnflag)
# Sorting Might be possible with tf.nn.top_k and tf.gather_nd with tf.meshgrid
def q1():
    shipdate = tf.placeholder(dtype=tf.int32, shape=(None,))
    returnflag = tf.placeholder(dtype=tf.string, shape=(None,))
    linestatus = tf.placeholder(dtype=tf.string, shape=(None,))
    quantity = tf.placeholder(dtype=tf.float32, shape=(None,))
    extendedprice = tf.placeholder(dtype=tf.float32, shape=(None,))
    discount = tf.placeholder(dtype=tf.float32, shape=(None,))
    tax = tf.placeholder(dtype=tf.float32, shape=(None,))
    zeros = tf.zeros_like(discount)
    ones = tf.ones_like(discount)
    minus_one = tf.constant(-1.0,dtype=tf.float32)

    R = tf.cast(tf.where(tf.equal(returnflag,'R'),ones,zeros),tf.bool)
    N = tf.cast(tf.where(tf.equal(returnflag,'N'),ones,zeros),tf.bool)
    A = tf.cast(tf.where(tf.equal(returnflag,'A'),ones,zeros),tf.bool)

    O = tf.cast(tf.where(tf.equal(linestatus,'O'),ones,zeros),tf.bool)
    F = tf.cast(tf.where(tf.equal(linestatus,'F'),ones,zeros),tf.bool)
    shipdate = tf.less_equal(shipdate, 19980901)

    # AF
    group_filter_1 = tf.logical_and(tf.logical_and(A,F),shipdate)
    # NF
    group_filter_2 = tf.logical_and(tf.logical_and(N,F),shipdate)
    # NO
    group_filter_3 = tf.logical_and(tf.logical_and(N,O),shipdate)
    # RF
    group_filter_4 = tf.logical_and(tf.logical_and(R,F),shipdate)

    #Group 1
    sum_qty_1 = tf.reduce_sum(tf.where(group_filter_1,quantity,zeros))
    sum_base_price_1 = tf.reduce_sum(tf.where(group_filter_1,extendedprice,zeros))
    sum_disc_price_1 = tf.reduce_sum(tf.where(group_filter_1,tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),zeros))
    sum_charge_1 = tf.reduce_sum(tf.where(group_filter_1, tf.multiply(tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),tf.add(ones,tax))
,zeros))
    count_1 = tf.reduce_sum(tf.cast(group_filter_1,tf.float32))
    avg_qty_1 = tf.div(tf.reduce_sum(tf.where(group_filter_1,quantity,zeros)),tf.reduce_sum(count_1)) #int32 is slower  
    avg_price_1 = tf.div(tf.reduce_sum(tf.where(group_filter_1,extendedprice,zeros)),tf.reduce_sum(count_1)) #int32 is slower  
    avg_disc_1 = tf.div(tf.reduce_sum(tf.where(group_filter_1,discount,zeros)),tf.reduce_sum(count_1)) #int32 is slower  
    
    #Group 2
    sum_qty_2 = tf.reduce_sum(tf.where(group_filter_2,quantity,zeros))
    sum_base_price_2 = tf.reduce_sum(tf.where(group_filter_2,extendedprice,zeros))
    sum_disc_price_2 = tf.reduce_sum(tf.where(group_filter_2,tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),zeros))
    sum_charge_2 = tf.reduce_sum(tf.where(group_filter_2, tf.multiply(tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),tf.add(ones,tax))
,zeros))
    count_2 = tf.reduce_sum(tf.cast(group_filter_2,tf.float32))
    avg_qty_2 = tf.div(tf.reduce_sum(tf.where(group_filter_2,quantity,zeros)),tf.reduce_sum(count_2)) #int32 is slower  
    avg_price_2 = tf.div(tf.reduce_sum(tf.where(group_filter_2,extendedprice,zeros)),tf.reduce_sum(count_2)) #int32 is slower  
    avg_disc_2 = tf.div(tf.reduce_sum(tf.where(group_filter_2,discount,zeros)),tf.reduce_sum(count_2)) #int32 is slower     

    #Group 3
    sum_qty_3 = tf.reduce_sum(tf.where(group_filter_3,quantity,zeros))
    sum_base_price_3 = tf.reduce_sum(tf.where(group_filter_3,extendedprice,zeros))
    sum_disc_price_3 = tf.reduce_sum(tf.where(group_filter_3,tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),zeros))
    sum_charge_3 = tf.reduce_sum(tf.where(group_filter_3, tf.multiply(tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),tf.add(ones,tax))
,zeros))
    count_3 = tf.reduce_sum(tf.cast(group_filter_3,tf.float32))
    avg_qty_3 = tf.div(tf.reduce_sum(tf.where(group_filter_3,quantity,zeros)),tf.reduce_sum(count_3)) #int32 is slower  
    avg_price_3 = tf.div(tf.reduce_sum(tf.where(group_filter_3,extendedprice,zeros)),tf.reduce_sum(count_3)) #int32 is slower  
    avg_disc_3 = tf.div(tf.reduce_sum(tf.where(group_filter_3,discount,zeros)),tf.reduce_sum(count_3)) #int32 is slower     

     #Group 4
    sum_qty_4 = tf.reduce_sum(tf.where(group_filter_4,quantity,zeros))
    sum_base_price_4 = tf.reduce_sum(tf.where(group_filter_4,extendedprice,zeros))
    sum_disc_price_4 = tf.reduce_sum(tf.where(group_filter_4,tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),zeros))
    sum_charge_4 = tf.reduce_sum(tf.where(group_filter_4, tf.multiply(tf.multiply(tf.add(ones,tf.multiply(minus_one,discount)),extendedprice),tf.add(ones,tax))
,zeros))
    count_4 = tf.reduce_sum(tf.cast(tf.logical_and(group_filter_4,shipdate),tf.float32))
    avg_qty_4 = tf.div(tf.reduce_sum(tf.where(group_filter_4,quantity,zeros)),tf.reduce_sum(count_4)) #int32 is slower  
    avg_price_4 = tf.div(tf.reduce_sum(tf.where(group_filter_4,extendedprice,zeros)),tf.reduce_sum(count_4)) #int32 is slower  
    avg_disc_4 = tf.div(tf.reduce_sum(tf.where(group_filter_4,discount,zeros)),tf.reduce_sum(count_4)) #int32 is slower     

    result = sum_qty_1,sum_base_price_1,sum_disc_price_1,sum_charge_1,avg_qty_1,avg_price_1,avg_disc_1,count_1,sum_qty_2,sum_base_price_2,sum_disc_price_2,sum_charge_2,avg_qty_2,avg_price_2,avg_disc_2,count_2,sum_qty_3,sum_base_price_3,sum_disc_price_3,sum_charge_3,avg_qty_3,avg_price_3,avg_disc_3,count_3,sum_qty_4,sum_base_price_4,sum_disc_price_4,sum_charge_4,avg_qty_4,avg_price_4,avg_disc_4,count_4
    
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
    # avg_qty
    # avg_price
    # avg_disc
    # count

def q6():
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
