import os

client_stmt = "psql -p 7483 -h localhost -U holanda -c "

def runMicro(scale):
    filterSQL  = '\"select l_quantity from lineitem_%d where l_quantity BETWEEN 10 AND 24;\"'%scale
    join_SQL = '\"select l_orderkey from lineitem_%d inner join orders_%d  on(l_orderkey = o_orderkey);\"'%(scale,scale)
    aggregationSQL = '\"select sum(l_quantity) from lineitem_%d;\"'%scale
    group_by_SQL = '\"select sum(l_quantity) from lineitem_%d group by l_returnflag;\"'%scale
    order_by_SQL = '\"select l_quantity from lineitem_%d order by l_quantity;\"'%scale
    for i in range (0,5):
        os.system(client_stmt+filterSQL)
    for i in range (0,5):    
        os.system(client_stmt+aggregationSQL)
    for i in range (0,5):   
        os.system(client_stmt+group_by_SQL)
    for i in range (0,5):
        os.system(client_stmt+order_by_SQL)
    for i in range (0,5):
          os.system(client_stmt+join_SQL)

runMicro(0.1)
runMicro(1)
runMicro(10)
# runTPCH(100)