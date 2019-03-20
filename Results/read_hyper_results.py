import json

query_time = []
queries = []
scales = [0,1,10]
i = 0
saveQueryEnd = False
query_sql = 0
execution_time = 0

for scale in scales:
    filterSQL  = 'select sum(l_quantity) from lineitem_%d where l_quantity BETWEEN 10 AND 24'%scale
    joinSQL = 'select sum(l_quantity) from lineitem_%d inner join orders_%d  on(l_orderkey = o_orderkey)'%(scale,scale)
    aggregationSQL = 'select l_returnflag,sum(l_quantity) from lineitem_%d group by l_returnflag'%scale
    limitSQL = 'select l_quantity from lineitem_%d order by l_quantity limit 10'%scale
    query_1  = 'select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order from lineitem_%d where l_shipdate <= date \'1998-12-01\' group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus'%scale
    query_6 = 'select sum(l_extendedprice * l_discount) as revenue from lineitem_%d where l_shipdate >= date \'1994-01-01\' and l_shipdate < date \'1995-01-01\' and l_discount between 0.06 - 0.01 and 0.06 + 0.01 and l_quantity < 24'%scale
    queries.append(filterSQL)
    queries.append(joinSQL)
    queries.append(aggregationSQL)
    queries.append(limitSQL)
    queries.append(query_1)
    queries.append(query_6)

for line in open('hyperd.log', 'r'):
	json_line = json.loads(line)
	for query in queries:
		if (json_line['k'] == "query-begin"):
			if(json_line['v']['query'] == query):
				print "test"

				i = i + 1
				if (i == 5):
					query_sql = query
					saveQueryEnd = True
	if(i == 5):
		if (json_line['k'] == "query-end"):
			i = 0
			execution_time = json_line['v']['execution-time']
			query_time.append([query_sql,execution_time])

print (query_time)