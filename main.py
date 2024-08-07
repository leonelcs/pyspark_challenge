from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, udf, lag, struct
from pyspark.sql.types import StructType, StructField, StringType, DateType
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder.appName("Group Transactions").getOrCreate()

schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("transaction_id", StringType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("checkin_date", DateType(), True),
    StructField("checkout_date", DateType(), True)
])

# df = spark.createDataFrame(data, schema=schema)
df = spark.read.csv(
    "data/transactions.csv", 
    header=True,
    mode="DROPMALFORMED", 
    schema=schema
)

# Define a window partitioned by user, ordered by checkin_date
window_spec = Window.partitionBy("user_id").orderBy("checkin_date")

# Define a UDF to determine if the current transaction should be grouped with the previous
def should_group(prev_checkin_string, prev_checkout_date, current_checkin_date):
    if prev_checkout_date is None:
        return 'trip_'+current_checkin_date.strftime("%m%d%Y")
    if (current_checkin_date - prev_checkout_date).days <= 3:
        return 'trip_'+prev_checkin_string.strftime("%m%d%Y")
    else:
        return 'trip_'+current_checkin_date.strftime("%m%d%Y")

group_udf = udf(should_group, StringType())

cancelled = df.filter(col("transaction_type") == "CANCELLATION")
others = df.filter(col("transaction_type") != "CANCELLATION")

valid = others.join(cancelled, on='transaction_id', how='left_anti')
valid.show()

# Add a column to determine groups
valid = valid.withColumn("prev_checkout_date", lag(df.checkout_date).over(window_spec))
valid = valid.withColumn("prev_checkin_string", lag(df.checkin_date).over(window_spec))
valid.show()
valid = valid.withColumn("trip_id", group_udf(col('prev_checkin_string'), col("prev_checkout_date"),col("checkin_date")))

# Group by user_id and trip_id, aggregate transactions into lists
grouped_df = valid.groupBy("user_id", "trip_id").agg(
    collect_list(struct("transaction_id","transaction_type", "checkin_date", "checkout_date")).alias("transactions")
).orderBy("user_id", "trip_id")

grouped_df.show(truncate=False)

# Stop the Spark session
spark.stop()