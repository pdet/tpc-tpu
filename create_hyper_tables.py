import os

client_stmt = "psql -p 7483 -h localhost -U pedroholanda -c "

def dropTables(scale):
	dropTable =  "\"DROP TABLE LINEITEM_%d;\""%scale
	os.system(client_stmt+dropTable)
	dropTable =  "\"DROP TABLE ORDERS_%d;\""%scale
	os.system(client_stmt+dropTable)
	dropTable =  "\"DROP TABLE SUPPLIER_%d;\""%scale
	os.system(client_stmt+dropTable)
	dropTable =  "\"DROP TABLE NATION_%d;\""%scale
	os.system(client_stmt+dropTable)

# dropTables(0.1)
# dropTables(1)
# dropTables(10)
# dropTables(100)

def loadTables(scale):
	tbl_dir = "/home/pedroholanda/tpch-"+str(scale)
	create_lineitem = '\"CREATE TABLE LINEITEM_%d(L_ORDERKEY    INTEGER NOT NULL, L_PARTKEY     INTEGER NOT NULL, L_SUPPKEY     INTEGER NOT NULL,L_LINENUMBER  INTEGER NOT NULL,L_QUANTITY    DECIMAL(15,2) NOT NULL, L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,  L_DISCOUNT    DECIMAL(15,2) NOT NULL, L_TAX         DECIMAL(15,2) NOT NULL, L_RETURNFLAG  CHAR(1) NOT NULL, L_LINESTATUS  CHAR(1) NOT NULL, L_SHIPDATE    DATE NOT NULL, L_COMMITDATE  DATE NOT NULL, L_RECEIPTDATE DATE NOT NULL, L_SHIPINSTRUCT CHAR(25) NOT NULL, L_SHIPMODE     CHAR(10) NOT NULL, L_COMMENT      VARCHAR(44) NOT NULL);\"'%scale
	create_orders = '\"CREATE TABLE ORDERS_%d( O_ORDERKEY       INTEGER NOT NULL, O_CUSTKEY        INTEGER NOT NULL, O_ORDERSTATUS    CHAR(1) NOT NULL, O_TOTALPRICE     DECIMAL(15,2) NOT NULL, O_ORDERDATE      DATE NOT NULL, O_ORDERPRIORITY  CHAR(15) NOT NULL, O_CLERK          CHAR(15) NOT NULL, O_SHIPPRIORITY   INTEGER NOT NULL, O_COMMENT        VARCHAR(79) NOT NULL);\"'%scale
	create_supplier = '\"CREATE TABLE SUPPLIER_%d ( S_SUPPKEY     INTEGER NOT NULL, S_NAME        CHAR(25) NOT NULL, S_ADDRESS     VARCHAR(40) NOT NULL, S_NATIONKEY   INTEGER NOT NULL, S_PHONE       CHAR(15) NOT NULL, S_ACCTBAL     DECIMAL(15,2) NOT NULL, S_COMMENT     VARCHAR(101) NOT NULL);\"'%scale
	create_nation = '\"CREATE TABLE NATION_%d( N_NATIONKEY  INTEGER NOT NULL, N_NAME       CHAR(25) NOT NULL, N_REGIONKEY  INTEGER NOT NULL, N_COMMENT    VARCHAR(152));\"'%scale

	
	

	os.system(client_stmt+create_lineitem)
	os.system(client_stmt+create_orders)
	copysql =  "\"COPY LINEITEM_%d FROM \'"%scale + tbl_dir+ "/lineitem.tbl\' (FORMAT csv, DELIMITER '|', HEADER false, NULL '');\""
	os.system(client_stmt+copysql)
	copysql =  "\"COPY ORDERS_%d FROM \'"%scale + tbl_dir+ "/orders.tbl\' (FORMAT csv, DELIMITER '|', HEADER false, NULL '');\""
	os.system(client_stmt+copysql)
	copysql =  "\"COPY SUPPLIER_%d FROM \'"%scale + tbl_dir+ "/supplier.tbl\' (FORMAT csv, DELIMITER '|', HEADER false, NULL '');\""
	os.system(client_stmt+copysql)
	copysql =  "\"COPY NATION_%d FROM \'"%scale + tbl_dir+ "/nation.tbl\' (FORMAT csv, DELIMITER '|', HEADER false, NULL '');\""
	os.system(client_stmt+copysql)

loadTables(0.1)
loadTables(1)
loadTables(10)
# runTPCH(100)