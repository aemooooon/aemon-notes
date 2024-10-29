---
title: 
draft: false
tags: 
date: 2024-08-20
---
# Spark 概念与原理

## 什么是 Spark?
- **Spark** 是一个通用的集群计算框架，专门为解决 Hadoop、Pig 和 Hive 等系统的已知局限而设计。
- Spark 相较这些系统，**底层更加高效**，使用更加便捷。
- Spark 提供了用于 **分布式计算**、**数据查询**、**统计建模** 和 **机器学习** 的模块。
- **实现语言**：Spark 实现于 **Java**，并且提供了 **Scala**、**Python**、**R**、**SparkSQL** 等跨平台接口。
- **核心特性**：Spark 尽量将数据保存在 **内存** 中，最大限度地提高计算速度。

## Resilient Distributed Dataset (RDD)
- **RDD** 是 Spark 中最基本的数据抽象，表示一个分布式且不可变的对象集合。
- **RDD** 被划分为多个 **分区**（Partitions），这些分区分布在集群中的多个工作节点上。
- **RDD** 提供对数据进行转换的接口（如：`map`、`reduce`、`reduceByKey`、`groupByKey` 等）。
- **RDD** 可以代表持久化存储中的数据（如 **HDFS**、Cassandra），或缓存中的数据（如内存或本地磁盘）。
- **RDD** 也可以是另一个 **RDD** 的转换结果，即通过一系列 **transform** 操作链产生的新 **RDD**。

### RDD 元数据
- **Partitioner**：定义数据如何划分成多个分区的逻辑。
- **Partitions**：数据的实际分区。
- **Dependencies**：前置 **RDD** 的依赖关系列表。
- **Compute**：描述 **RDD** 是如何从上游 **RDD** 进行转换的逻辑。
- **Preferred Locations**：每个分区的优先存储位置（如：内存、磁盘、网络存储）。

### 容错机制
- **RDD** 通过 **重新计算** 或 **缓存策略** 实现数据的自动恢复，保证在节点失效或缓存丢失时，数据依然可以被计算或恢复。

## Directed Acyclic Graph (DAG)
- **DAG** 是对数据进行的 **transform** 和 **action** 操作的有向无环图。它描述了从输入数据到输出结果的所有操作步骤。
- **节点（Node）**：表示 **RDD**。
- **边（Edge）**：表示对 **RDD** 的转换（**transform**）或操作（**action**）。
- **Acyclic**：无环性，保证没有循环依赖。
- **Direct**：所有的转换操作都必须按顺序向前推进，不能回退。

## Spark 的懒执行与缓存机制
- Spark 使用 **惰性求值（Lazy Evaluation）**。当我们定义一个 **RDD** 时，Spark 不会立即执行操作，而是在执行诸如 `collect` 或 `saveAsTextFile` 等 **action** 操作时，才会从头开始执行所有的 **transform** 操作链。
- **缓存机制**：Spark 可以将中间计算结果（即 **RDD**）缓存到内存中，避免在重复使用该 **RDD** 时重新计算，提升性能。

## Spark 应用程序流程

1. **读取输入数据**：应用程序通过 **SparkContext**（`sc.textFile()` 或 `read()`）从外部存储系统（如 **HDFS** 或 SQL Server）读取数据。这些数据会被分区，并被分配到集群的不同节点上。在这个阶段，数据仍然是存储在 **HDFS** 或其他存储系统中的。
   
2. **构建 RDD**：通过对数据进行多种 **transform** 操作（如 `map`、`reduce`）生成新的 **RDD**。这些 **RDD** 可以是原始数据的转换结果，每一步 **transform** 操作都在内存中进行，尽可能避免频繁的磁盘 I/O。此时，数据已经被载入到内存中，并在内存中完成相应的计算。

3. **生成执行计划（DAG）**：在调用第一个 **action** 操作之前，Spark 会构建一个完整的 **DAG**，即从数据源到结果的所有操作链。这个 **DAG** 包含了所有的 **transform** 和 **action** 操作。

4. **执行并生成输出**：
   - 当执行 **action** 操作（如 `collect`、`saveAsTextFile`）时，Spark 会开始从头计算所有 **transform** 操作。
   - 在执行过程中，数据会经过内存的多次转换（如 `map`、`reduce`），并可以选择性地缓存到内存中以备后续使用。
   - 如果有必要，计算后的数据会被保存回 **HDFS** 或其他持久化存储系统中。
   - 数据流动路径：**HDFS -> 内存 -> 数据格式转换 -> 内存或缓存 -> 最终保存到 HDFS 或其他存储**。

5. **缓存**：在需要时，Spark 可以将计算过程中的中间结果缓存到内存中（通过 `cache()` 或 `persist()`），从而避免重复计算。如果缓存内存不足，数据可以自动溢出到磁盘以提高效率。

## Spark 应用程序的组件

### Application
- 一个 **Application** 表示 Spark 中一个完整的应用实例。它包含一系列 **transform** 操作，并最终生成对 **RDD** 的 **action**。

### Job
- **Job** 是指 Spark 应用程序中执行的完整操作链，通常由多个 **transform** 操作和最终的 **action** 构成。

### Stage
- **Stage** 是 **Job** 中可以并行执行的操作单元。一个 **Stage** 包含可以在多个分区上并行处理的 **transform** 操作。

### Task
- **Task** 是在单个分区上执行的 **Stage** 操作。每个分区的数据会由一个 **Task** 处理。

## Spark 的任务调度与执行

- **任务调度**：Spark 根据 **DAG** 将 **Job** 拆分为多个 **Stage**，每个 **Stage** 在不同的分区上并行执行。
- **数据重分布（Shuffle）**：在执行 `reduceByKey`、`groupByKey` 等需要聚合的操作时，Spark 可能需要在节点之间重新分布数据，以确保相同的 **key** 被移动到相同的分区进行处理。

## Spark Catalyst Optimizer
- **Catalyst Optimizer** 是 Spark 用于优化 SQL API 和 DataFrame API 的查询优化器。它使用了高级编程语言特性，如 **基于树的模式匹配（tree-based pattern matching）**，使得查询执行更高效。
- **Stages**：
  - **Analysis**：解析查询中的关系，映射命名属性到输入，并处理类型推断和转换。
  - **Logical optimization**：应用基于规则的查询优化，比如将查询计划中的冗余操作删除或合并。
  - **Physical planning**：应用物理优化，例如 **最小化 Shuffle 操作**，以减少网络传输和磁盘读写的开销。
  - **Code generation**：生成并编译实际运行在每个工作节点上的 Java 字节码。

## Spark API 概述

- **RDD API**：Spark 的构建基础，所有操作最终都会在 **RDD** 上执行。
- **SQL API**：替代 Hive 的更高效的 SQL 查询接口，通过 **Catalyst Optimizer** 提高性能。
- **DataFrame API**：以 **Dataset[Row]** 形式表示无类型（untyped）数据，并且数据按列组织。
- **Dataset API**：表示强类型（typed）的结构化数据，使用 Scala 或 Java 的 case classes（即 `Dataset[T]`）。

## Spark UI 概述

- **Jobs**：显示作业的事件时间轴及其详细信息。
- **Stages**：展示任务的详细信息，包括输入、输出字节数，读写状态等。
- **Storage**：显示 **RDD** 的存储状态和详细信息。
- **Environment**：显示 Spark 和 Hadoop 的属性配置。
- **Executors**：展示每个执行器的概述。
- **SQL**：显示所有被编译并执行的 SQL 查询的详细信息。

# Spark 内置 API 函数分类及解释

## Transformations

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit

# 初始化 SparkSession 和示例 DataFrame
spark = SparkSession.builder.appName("example").getOrCreate()
data = [(1, 2), (3, 4), (5, None)]
df = spark.createDataFrame(data, ["a", "b"])

# withColumn: 创建新列，基于现有列的表达式
df = df.withColumn("sum", col("a") + col("b"))

# withColumnRenamed: 重命名现有列
df = df.withColumnRenamed("sum", "total_sum")

# select: 选择列并评估表达式
df = df.select("a", "total_sum")

# groupBy: 对数据进行分组
df_grouped = df.groupBy("a")

# agg: 在分组后的数据上计算聚合函数
df_agg = df_grouped.agg({"total_sum": "sum"})

# join: 基于指定条件与另一个 DataFrame 连接
df2 = spark.createDataFrame([(1, "x"), (3, "y")], ["a", "category"])
df_joined = df.join(df2, on="a", how="inner")

# crossJoin: 执行笛卡尔积
df_cross_joined = df.crossJoin(df2)

# na.drop: 删除包含 null 的行
df_cleaned = df.na.drop(subset=["b"])

# na.fill: 用指定值替换 null
df_filled = df.na.fill(value=0, subset=["b"])

# na.replace: 用指定值替换其他值
df_replaced = df.na.replace({None: 0}, subset=["b"])

# stat.approxQuantile: 计算近似分位数
quantiles = df.approxQuantile("a", [0.25, 0.5, 0.75], 0.01)

# stat.sampleBy: 基于分层随机采样
fractions = {1: 0.5, 3: 0.5}
df_sampled = df.sampleBy("a", fractions, seed=42)

# stat.corr: 计算相关性
correlation = df.stat.corr("a", "b")

# stat.cov: 计算协方差
covariance = df.stat.cov("a", "b")
```

## Actions

```python
# show: 打印 DataFrame 的前 N 行
df.show()

# head: 返回 DataFrame 的前 N 行
df_head = df.head(2)

# count: 返回 DataFrame 的行数
row_count = df.count()

# describe: 计算数值列的统计信息
df.describe().show()

# collect: 收集 DataFrame 所有行
collected_data = df.collect()

# toPandas: 将 DataFrame 转换为 pandas DataFrame
df_pandas = df.toPandas()

# toLocalIterator: 返回一个本地 Python 迭代器
local_iter = df.toLocalIterator()

# foreach: 对每一行应用函数
df.foreach(lambda row: print(row))

# foreachPartition: 对每个分区应用函数
df.foreachPartition(lambda partition: print(list(partition)))

# write.csv: 将 DataFrame 写入 CSV 文件
df.write.csv("/path/to/save")

# write.json: 将 DataFrame 写入 JSON 文件
df.write.json("/path/to/save")

# write.parquet: 将 DataFrame 写入 Parquet 文件
df.write.parquet("/path/to/save")

```
## Regular Functions

```python
from pyspark.sql.functions import expr, lit, struct, when, col, substring, split, upper, lower

# expr: 评估 SQL 表达式
df = df.withColumn("expr_column", expr("a + 1"))

# lit: 创建包含常量值的新列
df = df.withColumn("constant_column", lit(100))

# struct: 创建嵌套列结构
df = df.withColumn("nested_column", struct("a", "b"))

# when / otherwise: 布尔逻辑条件评估
df = df.withColumn("new_column", when(col("b").isNull(), lit(0)).otherwise(col("b")))

# coalesce: 返回第一个非 null 的值
df = df.withColumn("coalesced_column", coalesce(col("a"), col("b")))

# substring: 基于位置和长度提取子字符串
df = df.withColumn("substring_column", substring(col("a").cast("string"), 1, 2))

# split: 基于分隔符分割字符串
df = df.withColumn("split_column", split(col("a").cast("string"), ","))

# upper / lower: 将字符串转换为大写或小写
df = df.withColumn("upper_column", upper(col("a").cast("string")))
df = df.withColumn("lower_column", lower(col("a").cast("string")))

# to_date / to_timestamp: 将字符串转换为 DateType 或 TimestampType
df = df.withColumn("date_column", to_date(col("a").cast("string"), "yyyy-MM-dd"))
df = df.withColumn("timestamp_column", to_timestamp(col("a").cast("string"), "yyyy-MM-dd HH:mm:ss"))

# explode: 对数组的每个元素生成一行
df_array = spark.createDataFrame([([1, 2, 3],)], ["array_col"])
df_exploded = df_array.select(explode(col("array_col")))

# flatten: 将数组中的数组平铺为单一数组
df_array = spark.createDataFrame([([[1, 2], [3, 4]],)], ["array_col"])
df_flattened = df_array.select(flatten(col("array_col")))

```

## Aggregation Functions

```python
from pyspark.sql.functions import count, countDistinct, first, last, max, min, avg, stddev, sum, collect_list, collect_set

# Group by 用于聚合
df_grouped = df.groupBy("a")

# count: 计算组内项目数量
df_grouped.count().show()

# countDistinct: 计算不同项目的数量
df_grouped.agg(countDistinct("b")).show()

# first / last: 返回组内第一个/最后一个值
df_grouped.agg(first("b")).show()
df_grouped.agg(last("b")).show()

# max / min: 计算组内最大值/最小值
df_grouped.agg(max("b")).show()
df_grouped.agg(min("b")).show()

# avg: 计算组内平均值
df_grouped.agg(avg("b")).show()

# stddev: 计算组内标准偏差
df_grouped.agg(stddev("b")).show()

# sum: 计算组内总和
df_grouped.agg(sum("b")).show()

# collect_list: 将组内的项目收集为列表（允许重复）
df_grouped.agg(collect_list("b")).show()

# collect_set: 将组内的项目收集为集合（唯一）
df_grouped.agg(collect_set("b")).show()

```

## Window Functions

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, percent_rank, ntile, cume_dist, lag, lead

# 定义 Window 规范
window_spec = Window.partitionBy("a").orderBy("b")

# row_number: 返回窗口内的顺序编号
df = df.withColumn("row_number", row_number().over(window_spec))

# rank: 返回窗口内行的排名
df = df.withColumn("rank", rank().over(window_spec))

# dense_rank: 返回窗口内行的密集排名
df = df.withColumn("dense_rank", dense_rank().over(window_spec))

# percent_rank: 返回窗口内行的相对排名
df = df.withColumn("percent_rank", percent_rank().over(window_spec))

# ntile: 将窗口内行分为 n 组
df = df.withColumn("ntile", ntile(4).over(window_spec))

# cume_dist: 返回窗口内值的累积分布
df = df.withColumn("cume_dist", cume_dist().over(window_spec))

# lag: 返回当前行之前 n 行的值
df = df.withColumn("lag_value", lag("b", 1).over(window_spec))

# lead: 返回当前行之后 n 行的值
df = df.withColumn("lead_value", lead("b", 1).over(window_spec))

df.show()

```