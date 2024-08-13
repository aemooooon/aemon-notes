---
title: 
draft: false
tags: 
date: 2024-08-14
---
# PySpark SQL 操作函数总结

## 1. `expr`
- **功能**: 解析和执行 SQL 表达式。
- **用法**: 可以接受完整的或部分 SQL 表达式，并将其应用到 DataFrame 上。
- **示例**:
  ```python
  from pyspark.sql.functions import expr
  df.withColumn("new_column", expr("existing_column * 2"))
  ```

## 2. `col`
- **功能**: 用于引用 DataFrame 中的列。
- **用法**: 通常用于表达式中，可以配合其他函数使用。
- **示例**:
  ```python
  from pyspark.sql.functions import col
  df.select(col("column_name"))
  ```

## 3. `lit`
- **功能**: 创建一个常量列。
- **用法**: 在表达式中使用，通常用于添加固定值的列。
- **示例**:
  ```python
  from pyspark.sql.functions import lit
  df.withColumn("constant_column", lit(100))
  ```

## 4. `when`
- **功能**: 类似于 SQL 中的 CASE WHEN 语句，进行条件判断。
- **用法**: 配合 `.otherwise()` 使用，形成条件表达式。
- **示例**:
  ```python
  from pyspark.sql.functions import when
  df.withColumn("category", when(col("value") > 10, "High").otherwise("Low"))
  ```

## 5. `concat`
- **功能**: 将多个列或字符串拼接在一起。
- **用法**: 可用于将多个字符串列拼接成一个新的列。
- **示例**:
  ```python
  from pyspark.sql.functions import concat, lit
  df.withColumn("full_name", concat(col("first_name"), lit(" "), col("last_name")))
  ```

## 6. `substring`
- **功能**: 提取字符串的一部分。
- **用法**: 类似于 SQL 中的 `SUBSTRING` 函数。
- **示例**:
  ```python
  from pyspark.sql.functions import substring
  df.withColumn("sub_col", substring(col("full_column"), 1, 3))
  ```

## 7. `alias`
- **功能**: 给列或表达式设置别名。
- **用法**: 类似于 SQL 中的 `AS` 关键字。
- **示例**:
  ```python
  df.select(col("column_name").alias("new_name"))
  ```

## 8. `cast`
- **功能**: 将列的数据类型进行转换。
- **用法**: 类似于 SQL 中的 `CAST` 函数。
- **示例**:
  ```python
  from pyspark.sql.functions import col
  df.withColumn("int_column", col("string_column").cast("int"))
  ```

## 9. `asc` / `desc`
- **功能**: 指定列的排序方式。
- **用法**: `asc` 表示升序，`desc` 表示降序。
- **示例**:
  ```python
  from pyspark.sql.functions import col
  df.orderBy(col("column_name").asc())
  ```

## 10. `isnull` / `isnotnull`
- **功能**: 检查列值是否为空或非空。
- **用法**: 用于过滤或条件判断。
- **示例**:
  ```python
  from pyspark.sql.functions import col
  df.filter(col("column_name").isnull())
  ```

## 11. `coalesce`
- **功能**: 返回第一个非空值。
- **用法**: 类似于 SQL 中的 `COALESCE` 函数，用于处理缺失值。
- **示例**:
  ```python
  from pyspark.sql.functions import coalesce
  df.withColumn("filled_column", coalesce(col("column1"), col("column2"), lit(0)))
  ```

## 12. `greatest` / `least`
- **功能**: 返回多个列中的最大值或最小值。
- **用法**: 类似于 SQL 中的 `GREATEST` 和 `LEAST` 函数。
- **示例**:
  ```python
  from pyspark.sql.functions import greatest, least
  df.withColumn("max_value", greatest(col("col1"), col("col2")))
  ```

## 13. `round`
- **功能**: 对数值列进行四舍五入。
- **用法**: 类似于 SQL 中的 `ROUND` 函数。
- **示例**:
  ```python
  from pyspark.sql.functions import round
  df.withColumn("rounded_value", round(col("value_column"), 2))
  ```

## 14. `date_format`
- **功能**: 将日期列格式化为特定格式。
- **用法**: 类似于 SQL 中的 `DATE_FORMAT` 函数。
- **示例**:
  ```python
  from pyspark.sql.functions import date_format
  df.withColumn("formatted_date", date_format(col("date_column"), "yyyy-MM-dd"))
  ```

## 15. `datediff` / `add_months`
- **功能**: 计算日期差值或添加日期。
- **用法**: `datediff` 用于计算两个日期之间的差值；`add_months` 用于向日期添加月份。
- **示例**:
  ```python
  from pyspark.sql.functions import datediff, add_months
  df.withColumn("days_diff", datediff(col("end_date"), col("start_date")))
  df.withColumn("next_month", add_months(col("start_date"), 1))
  ```

## 16. `array` / `array_contains`
- **功能**: 处理数组类型列。
- **用法**: `array` 用于创建数组列，`array_contains` 用于判断数组是否包含特定元素。
- **示例**:
  ```python
  from pyspark.sql.functions import array, array_contains
  df.withColumn("array_column", array(col("col1"), col("col2")))
  df.filter(array_contains(col("array_column"), "value"))
  ```

## 17. `split`
- **功能**: 将字符串按指定分隔符拆分为数组。
- **用法**: 类似于 SQL 中的 `SPLIT` 函数。
- **示例**:
  ```python
  from pyspark.sql.functions import split
  df.withColumn("split_column", split(col("string_column"), ","))
  ```

## 18. `regexp_extract` / `regexp_replace`
- **功能**: 使用正则表达式从字符串中提取或替换内容。
- **用法**: `regexp_extract` 用于提取，`regexp_replace` 用于替换。
- **示例**:
  ```python
  from pyspark.sql.functions import regexp_extract, regexp_replace
  df.withColumn("extracted", regexp_extract(col("text_column"), r'\d+', 0))
  df.withColumn("replaced", regexp_replace(col("text_column"), r'\d+', 'number'))
  ```

## 19. `groupBy` 和 聚合函数
- **功能**: 对数据进行分组，并使用聚合函数进行汇总操作。
- **常用聚合函数**: `sum`, `avg`, `count`, `max`, `min`, `countDistinct`
- **示例**:
  ```python
  df.groupBy("group_column").agg(sum("value_column").alias("total_value"))
  ```
