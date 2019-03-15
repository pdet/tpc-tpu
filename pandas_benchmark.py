import pandas as pd
import numpy as np
import timeit, csv
import os

os.chdir('tpch-dbgen')

lineitem = pd.read_csv("lineitem.tbl", sep='|',
                       names=["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice",
                              "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate",
                              "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
                       dtype={'l_returnflag': 'category', 'l_linestatus': 'category'},
                       parse_dates=['l_shipdate', 'l_commitdate', 'l_receiptdate'])

orders = pd.read_csv("orders.tbl", sep='|',
                     names=["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
                            "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
                     dtype={'o_orderstatus': 'category', 'o_orderpriority': 'category'}, parse_dates=['o_orderdate'])

customer = pd.read_csv("customer.tbl", sep='|',
                       names=["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment",
                              "c_comment"], dtype={'c_mktsegment': 'category'})

region = pd.read_csv("region.tbl", sep='|', names=["r_regionkey", "r_name", "r_comment"])
nation = pd.read_csv("nation.tbl", sep='|', names=["n_nationkey", "n_name", "n_regionkey", "n_comment"])
supplier = pd.read_csv("supplier.tbl", sep='|',
                       names=["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"])


def udf_disc_price(extended, discount):
    return np.multiply(extended, np.subtract(1, discount))


def udf_charge(extended, discount, tax):
    return np.multiply(extended, np.multiply(np.subtract(1, discount), np.add(1, tax)))


def q1():
    df = \
        lineitem[
            ["l_shipdate", "l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount", "l_tax"]][
            (lineitem['l_shipdate'] <= '1998-09-01')]
    df['disc_price'] = udf_disc_price(df['l_extendedprice'], df['l_discount'])
    df['charge'] = udf_charge(df['l_extendedprice'], df['l_discount'], df['l_tax'])
    res = df.groupby(['l_returnflag', 'l_linestatus']) \
        .agg({'l_quantity': 'sum', 'l_extendedprice': 'sum', 'disc_price': 'sum', 'charge': 'sum',
              'l_quantity': 'mean', 'l_extendedprice': 'mean', 'l_discount': 'mean', 'l_shipdate': 'count'})
    # print(res)
    return res


def q5():
    nr = nation.merge(region[region.r_name == "ASIA"], left_on="n_regionkey", right_on="r_regionkey")[
        ["n_nationkey", "n_name"]]
    snr = supplier[["s_suppkey", "s_nationkey"]].merge(nr, left_on="s_nationkey", right_on="n_nationkey")[
        ["s_suppkey", "s_nationkey", "n_name"]]
    lsnr = lineitem[["l_suppkey", "l_orderkey", "l_extendedprice", "l_discount"]].merge(snr, left_on="l_suppkey",
                                                                                        right_on="s_suppkey")
    o = orders[["o_orderkey", "o_custkey", "o_orderdate"]][
        (orders.o_orderdate >= "1994-01-01") & (orders.o_orderdate < "1995-01-01")][["o_orderkey", "o_custkey"]]
    oc = o.merge(customer[["c_custkey", "c_nationkey"]], left_on="o_custkey", right_on="c_custkey")[
        ["o_orderkey", "c_nationkey"]]
    lsnroc = lsnr.merge(oc, left_on=["l_orderkey", "s_nationkey"], right_on=["o_orderkey", "c_nationkey"])[
        ["l_extendedprice", "l_discount", "n_name"]]
    lsnroc["volume"] = lsnroc.l_extendedprice * (1 - lsnroc.l_discount)
    res = lsnroc.groupby("n_name").agg({'volume': sum}).reset_index().sort_values("volume", ascending=False)
    # print res
    return res


def q6():
    l = lineitem[["l_extendedprice", "l_discount", "l_shipdate", "l_quantity"]][
        (lineitem.l_shipdate >= "1994-01-01") &
        (lineitem.l_shipdate < "1995-01-01") &
        (lineitem.l_discount >= 0.05) &
        (lineitem.l_discount <= 0.07) &
        (lineitem.l_quantity < 24)][["l_extendedprice", "l_discount"]]
    res = (l.l_extendedprice * l.l_discount).sum()
    # print(res)
    return res


##### 
n = 5

f = open("pandas.csv", 'w')
writer = csv.writer(f)


def bench(q):
    res = timeit.repeat("q%d()" % q, setup="from __main__ import q%d" % q, number=1, repeat=n)
    print(res[4])
    writer.writerow(["pandas", "%d" % q, "%f" % res[4]])
    f.flush()


# bench(1)
bench(5)
# bench(6)
