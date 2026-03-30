from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, when, sum as ssum, avg, expr, floor, round as sround, lit
from pyspark.sql.types import DoubleType, StringType

spark = SparkSession.builder.master("local[*]").appName("Compras").getOrCreate()

raw = """<cole aqui o CSV acima>""".strip().splitlines()
header = raw[0].split(';')
data = [r.split(';') for r in raw[1:]]

rows = [Row(produto=r[0],
            quantidade=float(r[1]),
            preco_unit=float(r[2])) for r in data]

df = spark.createDataFrame(rows)

# Total por linha e geral
df = df.withColumn("total", col("quantidade") * col("preco_unit"))
valor_total = df.agg(ssum("total").alias("valor_total")).first()["valor_total"]

# Contagem de itens (fração -> 1, inteiro -> o próprio inteiro)
cont_itens = (df
  .select(when(col("quantidade") == floor(col("quantidade")),
               col("quantidade").cast("bigint")).otherwise(lit(1)).alias("cnt"))
  .agg(ssum("cnt").alias("itens")).first()["itens"])

# Mais caro (unitário)
mais_caro = df.orderBy(col("preco_unit").desc()).limit(1)

# Exibe o plano lógico/físico formatado
df.orderBy(col("total").desc()).limit(5).explain(mode="formatted")

# Acesse a UI do Spark (quando disponível localmente)
print(spark.sparkContext.uiWebUrl)  # tipicamente http://localhost:4040

from pyspark.sql.functions import count

# 1. Mais barato
mais_barato = df.orderBy(col("preco_unit").asc()).limit(1)

# 2. Médias
media_simples = df.agg(avg("preco_unit").alias("media")).first()["media"]
media_pond = (df.selectExpr("(preco_unit * quantidade) as pxq").agg(ssum("pxq")).first()[0] /
              df.agg(ssum("quantidade")).first()[0])

# 3. > R$ 10,00
n_maior_10 = df.filter(col("preco_unit") > 10).count()

# 4. Categoria
df = df.withColumn(
    "categoria",
    when(col("preco_unit") < 5, "Barato")
    .when(col("preco_unit") <= 15, "Médio")
    .otherwise("Caro")
)

dist_categorias = (df.groupBy("categoria").agg(count(lit(1)).alias("n")).orderBy("categoria"))

# 5. Top-5 por contribuição
top5 = df.orderBy(col("total").desc()).select("produto","quantidade","preco_unit","total").limit(5)
