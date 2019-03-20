import os

client_stmt = "psql -p 7483 -h localhost -U pedroholanda -c "

def runTPCH(scale):
    query_1  = '\"select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order from lineitem_%d where l_shipdate <= date \'1998-12-01\' group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus;\"'%scale
    query_6 = '\"select sum(l_extendedprice * l_discount) as revenue from lineitem_%d where l_shipdate >= date \'1994-01-01\' and l_shipdate < date \'1995-01-01\' and l_discount between 0.06 - 0.01 and 0.06 + 0.01 and l_quantity < 24;\"'%scale
    for i in range (0,5):
        os.system(client_stmt+query_1)
    for i in range (0,5):    
        os.system(client_stmt+query_6)
        
runTPCH(0.1)
runTPCH(1)
runTPCH(10)
# runTPCH(100)