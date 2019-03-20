import os

client_stmt = "psql -p 7483 -h localhost -U pedroholanda -c "

def runMicro(scale):
    filterSQL  = '\"select sum(l_quantity) from lineitem_%d where l_quantity BETWEEN 10 AND 24;\"'%scale
    join_SQL = '\"select sum(l_quantity) from lineitem_%d inner join orders_%d  on(l_orderkey = o_orderkey);\"'%(scale,scale)
    aggregationSQL = '\"select l_returnflag,sum(l_quantity) from lineitem_%d group by l_returnflag;\"'%scale
    limit_SQL = '\"select l_quantity from lineitem_%d order by l_quantity limit 10;\"'%scale
    for i in range (0,5):
        os.system(client_stmt+filterSQL)
    for i in range (0,5):
          os.system(client_stmt+join_SQL)
    for i in range (0,5):    
        os.system(client_stmt+aggregationSQL)
    for i in range (0,5):
        os.system(client_stmt+limit_SQL)


runMicro(0.1)
runMicro(1)
runMicro(10)
# runTPCH(100)