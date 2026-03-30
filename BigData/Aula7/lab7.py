"""
  1) Quantidade total de itens comprados (soma de QUANTIDADE)
  2) Valor total da compra (soma de QUANTIDADE * PRECO UNIT.)
  3) Produto(s) mais caro(s) pelo preço unitário
"""

from decimal import Decimal, ROUND_HALF_UP
from pyspark.sql import SparkSession, functions as F, types as T

# Utils 
file_path = "/Users/akatsurada/Documents/INSPER/BigData/Aula7/supermercado (3).csv"

def _to_decimal(col, scale: int) -> F.Column:
    """
    Converte strings numéricas que possam usar vírgula ou ponto como separador decimal
    p/ DecimalType(18, scale).
    - Remove espaços
    - Troca ',' por '.'
    - Faz cast p/ Decimal 
    """
    return (
        F.regexp_replace(F.trim(col), ",", ".")
        .cast(T.DecimalType(18, scale))
    )

def _fmt_money(d: Decimal, scale: int = 2) -> str:
    """Formata Decimal como moeda com 'scale' casas decimais (default 2)."""
    if d is None:
        d = Decimal(0)
    quant = Decimal(1).scaleb(-scale)  
    return f"{d.quantize(quant, rounding=ROUND_HALF_UP):f}"


def _fmt_qty(d: Decimal) -> str:
    """
    Formata a qtd removendo zeros à direita quando possível"""
    if d is None:
        return "0"
    # normalize() pode gerar notação científica; format 'f' evita
    s = format(d.normalize(), "f")
    # Remove zeros à direita e ponto 
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("analise-supermercado")
        .getOrCreate()
    )

    # Ler como string 
    raw_schema = T.StructType([
        T.StructField("PRODUTO", T.StringType(), False),
        T.StructField("QUANTIDADE", T.StringType(), False),
        T.StructField("PRECO UNIT. (R$)", T.StringType(), False),
    ])

    df_raw = (
        spark.read
        .option("header", True)
        .option("sep", ";")
        .schema(raw_schema)
        .csv(file_path)
    )

    # Padroniza nomes de colunas e tipa como Decimal
    df = (
        df_raw
        .withColumnRenamed("PRODUTO", "produto")
        .withColumnRenamed("QUANTIDADE", "quantidade_str")
        .withColumnRenamed("PRECO UNIT. (R$)", "preco_unitario_str")
        .withColumn("quantidade", _to_decimal(F.col("quantidade_str"), scale=3))   
        .withColumn("preco_unitario", _to_decimal(F.col("preco_unitario_str"), scale=2))
        .drop("quantidade_str", "preco_unitario_str")
    )

    # 1) Qtd total de itens comprados 
    total_quantidade_row = df.agg(F.sum("quantidade").alias("total_quantidade")).first()
    total_quantidade = total_quantidade_row["total_quantidade"]  # aqui em decimal

    # 2) Valor total da compra 
    df_valorizado = df.withColumn("valor_item", F.col("quantidade") * F.col("preco_unitario"))
    valor_total_row = df_valorizado.agg(F.sum("valor_item").alias("valor_total")).first()
    valor_total = valor_total_row["valor_total"]  

    # 3) Produto(s) mais caro(s) por preço unitário
    max_preco_row = df.agg(F.max("preco_unitario").alias("max_preco")).first()
    max_preco = max_preco_row["max_preco"]

    produtos_mais_caros = (
        df.filter(F.col("preco_unitario") == F.lit(max_preco))
          .select("produto", "preco_unitario")
          .orderBy("produto")
    )

    # Prints das respostas
    print("\nResultados:")
    print(f"Quantidade total comprada: { _fmt_qty(total_quantidade) }")
    print(f"Valor total da compra (R$): { _fmt_money(valor_total, 2) }")
    print("Produto(s) mais caro(s) por preço unitário:")
    produtos_mais_caros.select(
        "produto",
        F.format_number("preco_unitario", 2).alias("preco_unitario_R$")
    ).show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
