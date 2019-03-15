import os

client_file = "psql -p 7483 -h localhost -U holanda -f "
client_stmt = "psql -p 7483 -h localhost -U holanda -c "

tables = ["lineitem","customer","orders","nation", "supplier", "region"]
queries = ["query_1.sql","query_5.sql","query_6.sql"]

def runTPCH(scale):
	tbl_dir = "/export/scratch1/home/holanda/TPC-H/tpch-"+str(scale)
	os.system(client_file+"create.sql")
	for table in tables:
		copysql =  "\"COPY "+table+" FROM \'" + tbl_dir+ "/"+table+".tbl\' (FORMAT csv, DELIMITER '|', HEADER false, NULL '');\""
		os.system(client_stmt+copysql)

	for query in queries:
		os.system(client_file+query)

	for table in tables:
		os.system(client_stmt+"\"DROP TABLE "+table+";\"")
	
# runTPCH(0.1)
runTPCH(1)
# runTPCH(10)
# runTPCH(100)



