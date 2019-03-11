import os

client_file = "psql -p 7483 -h localhost -U holanda -f "
client_stmt = "psql -p 7483 -h localhost -U holanda -c "

def runTPCH(scale):
	os.system(client_file+"create.sql")
	lineitem_path = "/export/scratch1/home/holanda/TPC-H/tpch-"+str(scale)
	copysql =  "\"COPY lineitem          FROM \'" + lineitem_path+ "/lineitem.tbl\'          (FORMAT csv, DELIMITER '|', HEADER false, NULL '');\""

	# copysql = "COPY lineitem FROM "+ lineitem_path+"/lineitem.tbl WITH DELIMITER AS \"|\";"
	os.system(client_stmt+copysql)
	os.system(client_file+"query_1.sql")
	os.system(client_file+"query_6.sql")
	os.system(client_stmt+"\"DROP TABLE lineitem;\"")

runTPCH(0.1)
runTPCH(1)
runTPCH(10)
runTPCH(100)



