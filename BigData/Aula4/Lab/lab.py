# -*- coding: utf-8 -*-
"""
Spark
- Lab 1: Estimar o valor de pi lançando pontos aleatórios em um quadrado
- Lab 2: Transformações simples 
- Lab 3: Reviews de filmes
- Lab 4: Supermercado
"""


from __future__ import annotations 
import os 
from decimal import Decimal, InvalidOperation, getcontext  
from random import Random 
from typing import Iterator, Iterable, Tuple 
import csv  

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F  
from pyspark.sql import types as T 


def build_spark(app_name: str = "labs-solutions",
                master: str | None = None,
                shuffle_partitions: int | None = None) -> SparkSession:
    """
    Spark Session.
    """
    # Construtor
    builder = SparkSession.builder.appName(app_name)

    # Master padrão (local)
    if master:
        builder = builder.master(master)

    # Obter ou criar a sessão (getOrCreate)
    spark = builder.getOrCreate()

    # Nível de log ajustado para WARN
    spark.sparkContext.setLogLevel("WARN")

    # Padrão para dividir tarefas em x partes
    if shuffle_partitions is None:
        shuffle_partitions = max(32, spark.sparkContext.defaultParallelism * 2)

    # Partições de shuffle
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))

    return spark


# Utilitários
def create_or_replace_view(df: DataFrame, name: str) -> None:
    """
    Dar a um DataFrame um nome curto para facilitar a execução de SQL sobre ele.
    """
    # Nome da tabela
    df.createOrReplaceTempView(name)

def _first_existing(paths: list[str]) -> str | None:
    """
    Pega a primeira path que existe de uma lista de candidatos ou None
    """
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

# Lab 1 – Estimar pi c/ pontos aleatórios 
def pi_rdd(sc, num_samples: int = 10_000_00, seed: int = 42) -> float:
    """
    Estima pi usando pontos aleatórios no estilo RDD.
    - 'Lança' muitos dardos minúsculos no quadrante inferior direito de um quadrado.
    - Contar quantos caem dentro da figura redonda (um quarto de círculo) desenhada
      dentro desse quadrado.
    - A fração que cai dentro, multiplicada por 4, estaria proxima de pi.
    """
    # Lista de posições 0..num_samples-1, entre os workers.
    # Não precisa realmente dos números; eles apenas representam "quantos dardos lançar"
    # e ajudam a plataforma a dividir a carga.
    base = sc.range(num_samples, numSlices=sc.defaultParallelism * 4)

    # Helper que roda em cada partição dessa lista.
    # "idx" é o número da partição. "it" são os itens nessa partição.
    def count_hits(idx: int, it: Iterator[int]) -> Iterator[int]:
        # Gerador de números aleatórios. O seed depende do índice da partição.
        # Assim, os resultados são reprodutíveis e partições diferentes não se repetem.
        rng = Random(seed + idx)

        # Qts itens tem nnessa partição
        # Contagem = util, os valores em si ñ importam
        n_local = sum(1 for _ in it)

        # Começa c/ zero dardos dentro da forma circular
        hits = 0

        # Para cada dardo planejado nesta partição:
        for _ in range(n_local):
            # Ponto aleatório dentro do quadrado [0, 1] x [0, 1].
            x = rng.random()
            y = rng.random()

            # Se o ponto estiver dentro do quarto de círculo (distância da origem <= 1),
            # +1 na contagem de acerto
            hits += 1 if (x * x + y * y) <= 1.0 else 0

        # returna a contagem desta partição como um stream de um único item.
        yield hits

    # Todas as partições executam o helper e somam todos os acertos.
    inside = base.mapPartitionsWithIndex(count_hits).sum()

    # A fração dentro da forma (inside / total), multiplicada por 4, é nossa estimativa de pi
    return 4.0 * inside / num_samples


def pi_df(spark: SparkSession, num_samples: int = 10_000_00, seed: int = 42) -> float:
    """
    Estima pi usando pontos aleatórios no estilo DataFrame.
    - Criar uma tabela com uma linha por lançamento de dardo.
    - Para cada linha, escolher dois números aleatórios (x e y) entre 0 e 1.
    - Verificar se (x, y) cai dentro do quarto de círculo.
    - Tirar a média desses checks (1 para dentro, 0 para fora).
    - Multiplicar essa média por 4 para obter a estimativa de π.
    """
    # Criar uma tabela de uma coluna com "num_samples" linhas: 0, 1, 2, ...
    df = spark.range(num_samples)

    # Para cada linha, cria 2 valores aleatórios e verfica se o ponto está dentro
    # Duas chamadas rand c/ seeds diferentes 
    hits = df.select(
        # True se o ponto tá dentro do quarto de círculo
        (F.pow(F.rand(seed), 2) + F.pow(F.rand(seed + 1), 2) <= 1.0)
        # Convertendo True/False para 1/0 
        .cast("int").alias("hit")
    )

    # Média de todas as marcas 1/0 (isto é a fração dentro).
    # Multiplica por 4 para converter o resultado de um quadrante para o círculo completo.
    pi_val = hits.agg((F.lit(4.0) * F.avg("hit")).alias("pi")).first()["pi"]

    return float(pi_val)


def pi_sql(spark: SparkSession, num_samples: int = 10_000_00) -> float:
    """
    Monte Carlo pi em SparkSQL
    """
    # Table c/ N-rows 
    spark.range(num_samples).createOrReplaceTempView("nums")

    # Compute random hits using a CTE, then aggregate
    row = spark.sql("""
        WITH hits AS (
            SELECT CAST((pow(rand(42), 2) + pow(rand(43), 2) <= 1.0) AS INT) AS hit
            FROM nums
        )
        SELECT 4.0 * AVG(hit) AS pi
        FROM hits
    """).first()

    return float(row["pi"])

# Lab 2 – Transformações simples
def lab2_rdd(sc):
    """
      - Criar números de 1..30
      - Subtrair 1 de cada um
      - Ver os resultados no driver
      - Contar quantos existem
      - Manter apenas os menores que 10
    """
    # Criar números de 1 a 30 (inclusive) e distribui pelos workers
    rdd = sc.parallelize(range(1, 31), numSlices=sc.defaultParallelism)

    # Para cada número, -1
    minus_one = rdd.map(lambda x: x - 1)

    # Lista inteira de volta
    collected = minus_one.collect()

    # Contar quantos números (deve ser 30)
    count = minus_one.count()

    # Manter os números menores que 10 
    filtered_lt10 = minus_one.filter(lambda x: x < 10).collect()

    # Retornar todas as partes como um pequeno dict 
    return {
        "collected": collected,
        "count": count,
        "filtered_lt10": filtered_lt10,
    }


def lab2_df(spark: SparkSession):
    """
    Mesmos passos de lab2_rdd, mas usando o estilo DataFrame.
    """
    # Tabela de uma coluna com valores 1..30.
    df = spark.createDataFrame([(i,) for i in range(1, 31)], schema=["value"])

    # Nova coluna subtraindo 1.
    df1 = df.select((F.col("value") - 1).alias("value_minus_one"))

    # Valores como lista 
    collected = [r.value_minus_one for r in df1.collect()]

    # Contar quantas linhas há (deve ser 30).
    count = df1.count()

    # Manter apenas as linhas em que o novo valor é menor que 10; trazer de volta.
    filtered = [r.value_minus_one for r in df1.filter(F.col("value_minus_one") < 10).collect()]

    # Retornar todas as partes 
    return {
        "collected": collected,
        "count": count,
        "filtered_lt10": filtered,
    }


def lab2_sql(spark: SparkSession):
    """
    Mesmos passos novamente, agora usando SQL.
    """
    # Criar a mesma tabela de uma coluna 
    df = spark.createDataFrame([(i,) for i in range(1, 31)], schema=["value"])

    # nome
    create_or_replace_view(df, "nums")

    # Usar SQL para subtrair 1 e trazer os resultados como lista.
    collected = [r.value_minus_one for r in spark.sql("SELECT value - 1 AS value_minus_one FROM nums").collect()]

    # Contar as linhas após a alteração.
    count = spark.sql("SELECT COUNT(*) AS c FROM (SELECT value - 1 v FROM nums)").first()["c"]

    # manter os números pequenos e trazê-los como lista.
    filtered = [r.v for r in spark.sql("SELECT value - 1 AS v FROM nums WHERE value - 1 < 10").collect()]

    # Retornar todas as partes.
    return {
        "collected": collected,
        "count": count,
        "filtered_lt10": filtered,
    }


# Lab 3 – Filmes
RATINGS_SCHEMA = T.StructType([
    T.StructField("userId", T.IntegerType(), nullable=False),
    T.StructField("movieId", T.IntegerType(), nullable=False),
    T.StructField("rating", T.DoubleType(), nullable=False),
    T.StructField("timestamp", T.LongType(), nullable=False),
])


def read_ratings_df(spark: SparkSession, ratings_path: str) -> DataFrame:
    """
    Carrega o csv
    """
    # Cabeçalho False
    df = (spark.read
          .option("header", True)
          .schema(RATINGS_SCHEMA)
          .csv(ratings_path))
    return df

def read_ratings_rdd(sc, ratings_path: str):
    """
    Carrega o arquivo usando RDD
    """

    # Partição de linhas
    def parse_partitions(lines: Iterator[str]) -> Iterator[Tuple[int, int, float, int]]:
        # Divide cada linha por vírgula, considerando aspas
        reader = csv.reader(lines)
        for row in reader:
            # Pular linhas vazias e a linha de cabeçalho que começa com 'userId'.
            if not row or row[0] == "userId":
                continue
            # Pega cada parte e casting
            uid = int(row[0]); mid = int(row[1]); rating = float(row[2]); ts = int(row[3])
            yield (uid, mid, rating, ts)

    # Devolve o stream de linhas parseadas
    return sc.textFile(ratings_path).mapPartitions(parse_partitions)

# A) Quantas linhas de avaliação existem no arquivo?
def lab3A_count_rdd(sc, ratings_path: str) -> int:
    # Ler o arquivo no estilo RDD
    rdd = read_ratings_rdd(sc, ratings_path)
    # Contar as linhas 
    return rdd.count()


def lab3A_count_df(spark: SparkSession, ratings_path: str) -> int:
    # Ler o arquivo no estilo DataFrame
    df = read_ratings_df(spark, ratings_path)
    # Conta as linhas e retorna
    return df.count()


def lab3A_count_sql(spark: SparkSession, ratings_path: str) -> int:
    # Ler o arquivo como tabela
    df = read_ratings_df(spark, ratings_path)
    # nome
    create_or_replace_view(df, "ratings")
    # Número de linhas
    return spark.sql("SELECT COUNT(*) AS c FROM ratings").first()["c"]


# --- B) Quantas avaliações cada filme recebeu?
def lab3B_rdd(sc, ratings_path: str):
    # Ler o arquivo como linhas de (usuário, filme, nota, tempo).
    rdd = read_ratings_rdd(sc, ratings_path)
    # P/ cada linha, manter apenas o movieId e o número 1 
    # Soma esses 1's para o mesmo movieId
    counts = rdd.map(lambda t: (t[1], 1)).reduceByKey(lambda a, b: a + b)
    # Devolver movieId e número de avaliações
    return counts

def lab3B_df(spark: SparkSession, ratings_path: str) -> DataFrame:
    # Le como DataFrame.
    df = read_ratings_df(spark, ratings_path)
    # Agrupar linhas por movieId e contar linhas 
    return df.groupBy("movieId").agg(F.count("*").alias("num_ratings"))


def lab3B_sql(spark: SparkSession, ratings_path: str) -> DataFrame:
    # Nome
    df = read_ratings_df(spark, ratings_path)
    create_or_replace_view(df, "ratings")
    # Contar linhas por movieId.
    return spark.sql("""
        SELECT movieId, COUNT(*) AS num_ratings
        FROM ratings
        GROUP BY movieId
    """)

# C) Para cada filme, qual a menor e a maior nota?
def lab3C_rdd(sc, ratings_path: str):
    # Ler linhas
    rdd = read_ratings_rdd(sc, ratings_path)
    # P/ cada linha, manter (movieId, rating).
    by_movie = rdd.map(lambda t: (t[1], t[2]))

    # (min, max) por filme. Começa cada filme com (+infinito, -infinito)
    # assim qq valor real vai substituir e usar como limite
    zero = (float("inf"), float("-inf"))

    # Ao ver um novo rating:
    # o menor entre (min_atual, valor) é o novo min
    # o maior entre (max_atual, valor) é o novo max
    seq = lambda acc, x: (min(acc[0], x), max(acc[1], x))

    # Quando duas partições têm (min, max), pega o menor min
    # e o maior max.
    comb = lambda a, b: (min(a[0], b[0]), max(a[1], b[1]))

    # Junta tudo p/ chegar no movieId -> min_rating, max_rating
    minmax_by_movie = by_movie.aggregateByKey(zero, seq, comb)

    # Menor nota global, escolhe o par com o menor rating
    min_global = by_movie.takeOrdered(1, key=lambda kv: kv[1])[0]

    # Maior nota global, pega o par com o maior rating
    max_global = by_movie.takeOrdered(1, key=lambda kv: -kv[1])[0]

    # Resultados por filme e min max globais.
    return minmax_by_movie, min_global, max_global

def lab3C_df(spark: SparkSession, ratings_path: str) -> Tuple[DataFrame, Tuple[int, float], Tuple[int, float]]:
    # Ler como DataFrame.
    df = read_ratings_df(spark, ratings_path)

    # Para cada movieId, pegar a menor e a maior nota.
    minmax = df.groupBy("movieId").agg(
        F.min("rating").alias("min_rating"),
        F.max("rating").alias("max_rating")
    )

    # Encontra a menor nota em toda a tabela, ordena por rating ascendente e pega a 1a linha.
    min_global_row = df.orderBy(F.col("rating").asc()).select("movieId", "rating").first()

    # Encontra a maior nota, ordena por rating descendente e pega a 1a linha.
    max_global_row = df.orderBy(F.col("rating").desc()).select("movieId", "rating").first()

    # Resultados e min max globais.
    return minmax, (min_global_row["movieId"], min_global_row["rating"]), (max_global_row["movieId"], max_global_row["rating"])


def lab3C_sql(spark: SparkSession, ratings_path: str) -> Tuple[DataFrame, Tuple[int, float], Tuple[int, float]]:
    # Nome
    df = read_ratings_df(spark, ratings_path)
    create_or_replace_view(df, "ratings")

    # Para cada movieId, p/ menor e maior nota.
    minmax = spark.sql("""
        SELECT movieId, MIN(rating) AS min_rating, MAX(rating) AS max_rating
        FROM ratings
        GROUP BY movieId
    """)

    # Min global, pega a linha com a menor nota.
    min_global = spark.sql("SELECT movieId, rating FROM ratings ORDER BY rating ASC LIMIT 1").first()

    # Max global, pega a linha com a maior nota.
    max_global = spark.sql("SELECT movieId, rating FROM ratings ORDER BY rating DESC LIMIT 1").first()

    # Resultados e min max globais.
    return minmax, (min_global["movieId"], min_global["rating"]), (max_global["movieId"], max_global["rating"])


# D) Para cada pessoa, quantos filmes ela avaliou?

def lab3D_rdd(sc, ratings_path: str):
    # Ler linhas
    rdd = read_ratings_rdd(sc, ratings_path)
    # P/ cada linha, manter userId, 1 e somar os 1 por usuário
    return rdd.map(lambda t: (t[0], 1)).reduceByKey(lambda a, b: a + b)


def lab3D_df(spark: SparkSession, ratings_path: str) -> DataFrame:
    # Linhas por userId
    df = read_ratings_df(spark, ratings_path)
    return df.groupBy("userId").agg(F.count("*").alias("num_movies"))


def lab3D_sql(spark: SparkSession, ratings_path: str) -> DataFrame:
    # Tabela e nome, conta linhas por userId.
    df = read_ratings_df(spark, ratings_path)
    create_or_replace_view(df, "ratings")
    return spark.sql("""
        SELECT userId, COUNT(*) AS num_movies
        FROM ratings
        GROUP BY userId
    """)


# E) Quais filmes têm pelo menos uma avaliação de 5 estrelas?

def lab3E_rdd(sc, ratings_path: str):
    # Ler as linhas e manter c/ rating = 5.0, tb manter o movieId
    # dedupication nos ids
    rdd = read_ratings_rdd(sc, ratings_path)
    return rdd.filter(lambda t: t[2] == 5.0).map(lambda t: t[1]).distinct()


def lab3E_df(spark: SparkSession, ratings_path: str) -> DataFrame:
    # Filtrar linhas c/ rating == 5 e manter apenas movieId, distinct
    df = read_ratings_df(spark, ratings_path)
    return df.filter(F.col("rating") == 5.0).select("movieId").distinct()


def lab3E_sql(spark: SparkSession, ratings_path: str) -> DataFrame:
    # O mesmo em SQL.
    df = read_ratings_df(spark, ratings_path)
    create_or_replace_view(df, "ratings")
    return spark.sql("""
        SELECT DISTINCT movieId
        FROM ratings
        WHERE rating = 5.0
    """)


# F) Para cada filme, qual é a nota média?
def lab3F_rdd(sc, ratings_path: str):
    # Ler linhas
    rdd = read_ratings_rdd(sc, ratings_path)
    # movieId, (valor_da_nota, 1 avaliação
    pairs = rdd.map(lambda t: (t[1], (t[2], 1)))
    # Soma todos os valores de nota  e contagens por filme
    sums = pairs.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    # Total do valor dividido pela contagem (média)
    avg = sums.mapValues(lambda s: s[0] / s[1])
    # movieId, média
    return avg

def lab3F_df(spark: SparkSession, ratings_path: str) -> DataFrame:
    # Ler e calcular a média por movieId
    df = read_ratings_df(spark, ratings_path)
    return df.groupBy("movieId").agg(F.avg("rating").alias("avg_rating"))


def lab3F_sql(spark: SparkSession, ratings_path: str) -> DataFrame:
    # O mesmo em SQL.
    df = read_ratings_df(spark, ratings_path)
    create_or_replace_view(df, "ratings")
    return spark.sql("""
        SELECT movieId, AVG(rating) AS avg_rating
        FROM ratings
        GROUP BY movieId
    """)

# Lab4 Supermercado (três abordagens)

# Precisão
getcontext().prec = 28

#   nome do produto (texto), quantidade (número decimal com até 3 casas),
#   preço unitário (número decimal com 2 casas, como em moedas comuns).
SUPERMERCADO_SCHEMA = T.StructType([
    T.StructField("produto", T.StringType(), nullable=False),
    T.StructField("quantidade", T.DecimalType(18, 3), nullable=False),
    T.StructField("preco_unitario", T.DecimalType(18, 2), nullable=False),
])

def _parse_decimal(s: str) -> Decimal:
    """
    Converter um texto como '12.30' ou '12,30' em Decimal
    """
    # Remover espaços extras e trocar vírgula por ponto
    s = s.strip().replace(",", ".")
    try:
        # Converter texto em Decimal 
        return Decimal(s)
    except InvalidOperation:
        # Se o texto não for um número, raise error
        raise ValueError(f"Invalid decimal: {s}")

def supermercado_rdd(sc, compras_path: str):
    """
    Em RDD:
      - Qual é o custo total da compra?
      - Quantos itens foram comprados? Se a quantidade for fracionária, conta como 1
      - Qual produto tem o maior preço unitário?
    """
    # Ler o arquivo em linhas
    lines = sc.textFile(compras_path)

    # remove cabeçalho (a 1a linha com as cols)
    # Roda em cada partição de linhas, na 1a pula a primeira linha.
    def skip_header(idx: int, it: Iterator[str]) -> Iterator[str]:
        it = iter(it)
        if idx == 0:
            # Descartar a primeira linha na primeira partição
            next(it, None)
        for line in it:
            # Se tiver alguma linha tipo o cabeçalho, pular tb
            if line and not line.lower().startswith("produto;"):
                yield line

    # Remove o cabeçalho
    no_header = lines.mapPartitionsWithIndex(skip_header)

    # Dividir cada linha por ';' converte o text em Decimal
    def parse_row(line: str) -> Tuple[str, Decimal, Decimal]:
        # Separar pelo ponto e vírgula e limpar espaços
        parts = [p.strip() for p in line.split(";")]
        # 3 colunas
        if len(parts) != 3:
            raise ValueError(f"Invalid row (expected 3 columns): {line}")
        # Pegar cada parte e cast
        produto = parts[0]
        qtd = _parse_decimal(parts[1])
        preco = _parse_decimal(parts[2])
     
        return produto, qtd, preco

    # Stream de linhas 
    items = no_header.map(parse_row)  # (produto, quantidade, preco_unitario)

    # Se a quantidade for inteira, soma essa contagem exata; caso contrário, soma 1
    def item_count(qtd: Decimal) -> int:
        return int(qtd) if qtd == qtd.to_integral_value() else 1

    # Somar as contagens de itens em todas as linhas.
    num_produtos = items.map(lambda t: item_count(t[1])).sum()

    # Total: para cada linha, multiplica a qtd por preço 
    total = items.map(lambda t: (t[1] * t[2])).sum()

    # Linha c/ maior preço unitário.
    # Linha do topo em ordem decrescente pelo preço unit
    produto_mais_caro_row = items.takeOrdered(1, key=lambda t: -float(t[2]))[0]
    produto_mais_caro, preco_mais_caro = produto_mais_caro_row[0], produto_mais_caro_row[2]

    return {
        "num_produtos": float(num_produtos),  
        "total_compra": float(total),
        "produto_mais_caro": (produto_mais_caro, float(preco_mais_caro)),
    }


def supermercado_df(spark: SparkSession, compras_path: str):
    """
    Estilo DataFrame para as mesmas três perguntas.
    """
    # Ler o arquivo como uma tabela com três colunas, usando ';' como separador.
    df = (spark.read
          .option("header", True)
          .option("delimiter", ";")
          .schema(SUPERMERCADO_SCHEMA)
          .csv(compras_path))

    # Contar itens: se a quantidade não tiver casas decimais, contar esse valor; caso contrário, contar 1.
    num_produtos = (df
                    .select(F.when(F.col("quantidade") == F.floor("quantidade"),
                                   F.col("quantidade").cast("long"))
                            .otherwise(F.lit(1)).alias("cnt"))
                    .agg(F.sum("cnt").alias("num_produtos"))
                    .first()["num_produtos"])

    # Custo total: quantidade * preço unitário para cada linha e depois somar tudo.
    total_compra = (df
                    .select((F.col("quantidade") * F.col("preco_unitario")).alias("linha"))
                    .agg(F.sum("linha").alias("total"))
                    .first()["total"])

    # Produto mais caro por preço unitário: ordenar por preço desc e pegar a primeira linha.
    top = df.orderBy(F.col("preco_unitario").desc()).select("produto", "preco_unitario").first()

    # Retornar respostas organizadas.
    return {
        "num_produtos": int(num_produtos),
        "total_compra": float(total_compra),
        "produto_mais_caro": (top["produto"], float(top["preco_unitario"])),
    }


def supermercado_sql(spark: SparkSession, compras_path: str):
    """
    Mesmas três perguntas, agora em SQL.
    """
    # Ler a tabela e nomeá-la.
    df = (spark.read
          .option("header", True)
          .option("delimiter", ";")
          .schema(SUPERMERCADO_SCHEMA)
          .csv(compras_path))
    create_or_replace_view(df, "compras")

    # Contagem de itens com a mesma regra (inteiro conta a si próprio; senão conta 1).
    num_produtos = spark.sql("""
        SELECT SUM(CASE WHEN quantidade = floor(quantidade) THEN CAST(quantidade AS BIGINT) ELSE 1 END) AS num_produtos
        FROM compras
    """).first()["num_produtos"]

    # Regra do custo total.
    total_compra = spark.sql("""
        SELECT SUM(quantidade * preco_unitario) AS total
        FROM compras
    """).first()["total"]

    # Produto mais caro por preço unitário.
    top = spark.sql("""
        SELECT produto, preco_unitario
        FROM compras
        ORDER BY preco_unitario DESC
        LIMIT 1
    """).first()

    # Retornar respostas.
    return {
        "num_produtos": int(num_produtos),
        "total_compra": float(total_compra),
        "produto_mais_caro": (top["produto"], float(top["preco_unitario"])),
    }


# Main c/ td


def main():
    # --- Build Spark ---
    spark = build_spark(app_name="labs-solutions", master="local[*]")

    print("\n==================== Lab 1 — Monte Carlo π ====================")
    print(f"RDD      π ≈ {pi_rdd(spark.sparkContext, num_samples=1_000_000):.5f}")
    print(f"DataFrame π ≈ {pi_df(spark, num_samples=1_000_000):.5f}")
    print(f"SQL       π ≈ {pi_sql(spark, num_samples=1_000_000):.5f}")

    print("\n==================== Lab 2 — Simple transforms =================")
    lab2_r = lab2_rdd(spark.sparkContext)
    lab2_d = lab2_df(spark)
    lab2_s = lab2_sql(spark)
    print(f"RDD: count={lab2_r['count']}, <10={lab2_r['filtered_lt10']}")
    print(f"DF : count={lab2_d['count']}, <10={lab2_d['filtered_lt10']}")
    print(f"SQL: count={lab2_s['count']}, <10={lab2_s['filtered_lt10']}")

    print("\n==================== Lab 3 — MovieLens ratings =================")

    # Allow user override via CLI/env; otherwise try common defaults seen in your paths
    ratings_candidates = [
        os.environ.get("RATINGS_CSV"),
        "/Users/akatsurada/Documents/INSPER/BigData/Aula4/Lab/ratings.csv",
        "/home/pads/notebooks/PADSONL07/dados/aula 04/ml-25m/ratings.csv",
    ]
    ratings_path = _first_existing(ratings_candidates)

    if ratings_path is None:
        print("⚠️  ratings.csv not found. Set RATINGS_CSV env var or adjust the defaults in main(). Skipping Lab 3.")
    else:
        print(f"Using ratings file: {ratings_path}")

# --- replace the Lab 3 printing block in main() with the following ---
        # A) count
        print("\nA) Count ratings (número total de linhas)")
        a_rdd = lab3A_count_rdd(spark.sparkContext, ratings_path)
        a_df  = lab3A_count_df(spark, ratings_path)
        a_sql = lab3A_count_sql(spark, ratings_path)
        print(f"  RDD: {a_rdd}")
        print(f"  DF : {a_df}")
        print(f"  SQL: {a_sql}")

        # B) ratings per movie (multi-row answer)
        print("\nB) Ratings per movie — 'quantas avaliações cada filme recebeu'")
        b_rdd = lab3B_rdd(spark.sparkContext, ratings_path)
        b_df  = lab3B_df(spark, ratings_path)
        b_sql = lab3B_sql(spark, ratings_path)
        print(f"  RDD total filmes = {b_rdd.count()}  | mostrando top 10 por contagem:")
        print("   ", b_rdd.takeOrdered(10, key=lambda kv: -kv[1]))
        print(f"  DF  total filmes = {b_df.count()}  | mostrando top 10:")
        b_df.orderBy(F.desc("num_ratings")).show(10, truncate=False)
        print(f"  SQL total filmes = {b_sql.count()} | mostrando top 10:")
        b_sql.orderBy(F.desc("num_ratings")).show(10, truncate=False)

        # C) min/max por filme + extremos globais (pergunta original sugere ambos)
        print("\nC) Menor e maior nota por filme + extremos globais")
        c_rdd_mm, c_rdd_min, c_rdd_max = lab3C_rdd(spark.sparkContext, ratings_path)
        print(f"  RDD extremos globais  -> min: {c_rdd_min} | max: {c_rdd_max}")
        print(f"  RDD linhas por-filme  = {c_rdd_mm.count()} | amostra 10:")
        print("   ", c_rdd_mm.take(10))

        c_df_mm, c_df_min, c_df_max = lab3C_df(spark, ratings_path)
        print(f"  DF  extremos globais  -> min: {c_df_min} | max: {c_df_max}")
        print(f"  DF  linhas por-filme  = {c_df_mm.count()} | amostra 10:")
        c_df_mm.orderBy("movieId").show(10, truncate=False)

        c_sql_mm, c_sql_min, c_sql_max = lab3C_sql(spark, ratings_path)
        print(f"  SQL extremos globais  -> min: {c_sql_min} | max: {c_sql_max}")
        print(f"  SQL linhas por-filme  = {c_sql_mm.count()} | amostra 10:")
        c_sql_mm.orderBy("movieId").show(10, truncate=False)

        # D) número de filmes avaliados por usuário (multi-row answer)
        print("\nD) Quantos filmes cada usuário avaliou")
        d_rdd = lab3D_rdd(spark.sparkContext, ratings_path)
        d_df  = lab3D_df(spark, ratings_path)
        d_sql = lab3D_sql(spark, ratings_path)
        print(f"  RDD total usuários = {d_rdd.count()} | top 10 por contagem:")
        print("   ", d_rdd.takeOrdered(10, key=lambda kv: -kv[1]))
        print(f"  DF  total usuários = {d_df.count()} | top 10:")
        d_df.orderBy(F.desc("num_movies")).show(10, truncate=False)
        print(f"  SQL total usuários = {d_sql.count()} | top 10:")
        d_sql.orderBy(F.desc("num_movies")).show(10, truncate=False)

        # E) filmes com ao menos uma avaliação 5 estrelas (multi-row answer)
        print("\nE) Filmes com exatamente 5 estrelas (distintos)")
        e_rdd = lab3E_rdd(spark.sparkContext, ratings_path)
        e_df  = lab3E_df(spark, ratings_path)
        e_sql = lab3E_sql(spark, ratings_path)
        print(f"  RDD total filmes = {e_rdd.count()} | primeiros 10 ids: {e_rdd.take(10)}")
        print(f"  DF  total filmes = {e_df.count()} | amostra 10:")
        e_df.show(10, truncate=False)
        print(f"  SQL total filmes = {e_sql.count()} | amostra 10:")
        e_sql.show(10, truncate=False)

        # F) nota média por filme (multi-row answer)
        print("\nF) Nota média por filme")
        f_rdd = lab3F_rdd(spark.sparkContext, ratings_path)
        f_df  = lab3F_df(spark, ratings_path)
        f_sql = lab3F_sql(spark, ratings_path)
        print(f"  RDD total filmes = {f_rdd.count()} | top 10 por média:")
        print("   ", f_rdd.takeOrdered(10, key=lambda kv: -kv[1]))
        print(f"  DF  total filmes = {f_df.count()} | top 10 por média:")
        f_df.orderBy(F.desc("avg_rating")).show(10, truncate=False)
        print(f"  SQL total filmes = {f_sql.count()} | top 10 por média:")
        f_sql.orderBy(F.desc("avg_rating")).show(10, truncate=False)


    print("\n==================== Prática — Supermercado ====================")

    supermercado_candidates = [
        os.environ.get("COMPRAS_CSV"),
        "/Users/akatsurada/Documents/INSPER/BigData/Aula4/Lab/supermercado (3).csv",
    ]
    compras_path = _first_existing(supermercado_candidates)

    if compras_path is None:
        print("⚠️  supermercado.csv not found. Set COMPRAS_CSV env var or adjust the defaults in main(). Skipping Supermercado.")
    else:
        print(f"Using supermarket file: {compras_path}")
        rdd_res = supermercado_rdd(spark.sparkContext, compras_path)
        df_res  = supermercado_df(spark, compras_path)
        sql_res = supermercado_sql(spark, compras_path)

        print("\nRDD:", rdd_res)
        print("DF :", df_res)
        print("SQL:", sql_res)

    # Tidy shutdown
    spark.stop()


if __name__ == "__main__":
    main()