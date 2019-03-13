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
# l_returnflag = l_returnflag[l_returnflag =="R"] = "1"
# l_returnflag = l_returnflag[l_returnflag =="N"] = "2"
# l_returnflag = l_returnflag[l_returnflag =="A"] = "3"
# l_returnflag = l_returnflag.astype('int32')
l_linestatus = lineitem_df["l_returnflag"].values.astype('S1')
# l_linestatus = l_linestatus[l_linestatus =="O"] = "1"
# l_linestatus = l_linestatus[l_linestatus =="F"] = "2"
# l_linestatus = l_linestatus.astype('int32')
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

# select
#         l_returnflag,
#         l_linestatus,
#         sum(l_quantity) as sum_qty,
#         sum(l_extendedprice) as sum_base_price,
#         sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
#         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
#         avg(l_quantity) as avg_qty,
#         avg(l_extendedprice) as avg_price,
#         avg(l_discount) as avg_disc,
#         count(*) as count_order
# from
#         lineitem
# where
#         l_shipdate <= date '1998-12-01' - interval '90' day
# group by
#         l_returnflag,
#         l_linestatus
# order by
#         l_returnflag,
#         l_linestatus;
    # lineitem[["l_shipdate", "l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount", "l_tax"]][
    #     (lineitem['l_shipdate'] <= '1998-09-01')]
    # df['disc_price'] = udf_disc_price(df['l_extendedprice'], df['l_discount'])
    # df['charge'] = udf_charge(df['l_extendedprice'], df['l_discount'], df['l_tax'])
    # return df.groupby(['l_returnflag', 'l_linestatus']) \
    #     .agg({'l_quantity': 'sum', 'l_extendedprice': 'sum', 'disc_price': 'sum', 'charge': 'sum',
    #           'l_quantity': 'mean', 'l_extendedprice': 'mean', 'l_discount': 'mean', 'l_shipdate': 'count'})

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
    left_group_1 = tf.where(tf.equal(returnflag,'R'),ones,zeros)
    left_group_2 = tf.where(tf.equal(returnflag,'N'),ones,zeros)
    left_group_3 = tf.where(tf.equal(returnflag,'A'),ones,zeros)
    right_group_1 = tf.where(tf.equal(returnflag,'O'),ones,zeros)
    right_group_2 = tf.where(tf.equal(returnflag,'F'),ones,zeros)
    # AF
    group_filter_1 = tf.logical_and(left_group_3,right_group_2)
    # NF
    group_filter_2 = tf.logical_and(left_group_2,right_group_2)
    # NO
    group_filter_3 = tf.logical_and(left_group_2,right_group_1)
    # RF
    group_filter_4 = tf.logical_and(left_group_1,right_group_2)
    shipdate = tf.less_equal(shipdate, 19980901)
    sum_qty = tf.reduce_sum(tf.where(tf.logical_and(shipdate,group_filter_1),quantity,zeros))
    sum_base_price = tf.reduce_sum(tf.where(tf.logical_and(shipdate,group_filter_1),extendedprice,zeros))
    sum_disc_price = tf.reduce_sum(tf.where(tf.logical_and(shipdate,group_filter_1),tf.multiply(tf.add(ones,discount),extendedprice),zeros))
    sum_charge = tf.reduce_sum(tf.where(tf.logical_and(shipdate,group_filter_1), tf.multiply(tf.multiply(tf.add(-1,discount),extendedprice),tf.add(ones,tax))
,zeros))
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
