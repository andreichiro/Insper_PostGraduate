from __future__ import annotations

import csv
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, List, Sequence, Tuple

from pyspark import RDD
from pyspark.storagelevel import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.functions import (
    col,
    coalesce,
    count,
    greatest,
    lit,
    mean,
    min as spark_min,
    max as spark_max,
    round as spark_round,
    stddev,
    sum as spark_sum,
    to_date,
    when,
)

import matplotlib.pyplot as plt
import matplotlib

DEFAULT_PATH_2009 = "/Users/akatsurada/Documents/INSPER/BigData/checkpoint/2009.csv"
DEFAULT_PATH_2011 = "/Users/akatsurada/Documents/INSPER/BigData/checkpoint/2011.csv"
DEFAULT_OUTPUT_PLOT = "./daily_flights.png"
DEFAULT_OUTPUT_MD = "./spark_answers.md"

DEFAULT_SPARK_MASTER = "local[2]"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SPARK_LOG_LEVEL = "ERROR"

DEFAULT_SQL_DEBUG_MAX_TO_STRING_FIELDS = "2000"

PATH_2009 = os.getenv("FLIGHTS_2009_PATH", DEFAULT_PATH_2009)
PATH_2011 = os.getenv("FLIGHTS_2011_PATH", DEFAULT_PATH_2011)
OUTPUT_PLOT_PATH = os.getenv("OUTPUT_PLOT_PATH", DEFAULT_OUTPUT_PLOT)
OUTPUT_MD_PATH = os.getenv("OUTPUT_MD_PATH", DEFAULT_OUTPUT_MD)
SPARK_MASTER = os.getenv("SPARK_MASTER", DEFAULT_SPARK_MASTER)
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
SPARK_LOG_LEVEL = os.getenv("SPARK_LOG_LEVEL", DEFAULT_SPARK_LOG_LEVEL)

logger = logging.getLogger("spark_flights_solution")


@dataclass
class MarkdownReport:
    path: str
    lines: List[str] = field(default_factory=list)

    def h(self, level: int, text: str) -> None:
        level = max(1, min(6, level))
        self.lines.append(f"{'#' * level} {text}\n")

    def p(self, text: str) -> None:
        self.lines.append(text.rstrip() + "\n\n")

    def code(self, text: str, language: str = "") -> None:
        self.lines.append(f"```{language}\n{text.rstrip()}\n```\n\n")

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        def esc(x: object) -> str:
            s = "" if x is None else str(x)
            return s.replace("\n", " ").replace("|", "\\|")

        self.lines.append("| " + " | ".join(map(esc, headers)) + " |\n")
        self.lines.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in rows:
            self.lines.append("| " + " | ".join(esc(v) for v in r) + " |\n")
        self.lines.append("\n")

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(self.lines)


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )


def banner(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def print_rdd_head(title: str, items: List[object], n: int = 10) -> str:
    banner(f"{title} (primeiras {n} linhas)")
    out_lines: List[str] = []
    for i, item in enumerate(items[:n], 1):
        line = f"{i:02d}: {item}"
        print(line)
        out_lines.append(line)
    return "\n".join(out_lines)


def df_show_string(df: DataFrame, n: int = 10, truncate: bool = False, vertical: bool = False) -> str:
    int_truncate = 0 if not truncate else 20
    return df._jdf.showString(n, int_truncate, vertical)


def show_df_head(title: str, df: DataFrame, n: int = 10) -> str:
    banner(f"{title} (primeiras {n} linhas)")
    s = df_show_string(df, n=n, truncate=False, vertical=False)
    print(s)
    return s


def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("FlightsRDD-DF-SQL-Solution")
        .master(SPARK_MASTER)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.files.maxPartitionBytes", str(256 * 1024 * 1024))
        .config("spark.sql.debug.maxToStringFields", DEFAULT_SQL_DEBUG_MAX_TO_STRING_FIELDS)
        .getOrCreate()
    )
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.sparkContext.setLogLevel(SPARK_LOG_LEVEL)
    return spark


def is_header_line(line: str) -> bool:
    return line.startswith("FL_DATE,OP_CARRIER,OP_CARRIER_FL_NUM,")


def parse_csv_partition(lines: Iterator[str]) -> Iterator[List[str]]:
    reader = csv.reader(lines)
    for row in reader:
        yield row


def safe_float(x: str) -> float | None:
    try:
        x = (x or "").strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def cancelled_flag(x: str) -> bool:
    v = safe_float(x)
    return bool(v is not None and v >= 0.5)


def read_2009_slim_rdd(sc, path: str) -> Tuple[List[str], RDD[Tuple[str, str, str, str]]]:
    raw: RDD[str] = sc.textFile(path)
    header_line = raw.first()
    header = next(csv.reader([header_line]))
    idx = {name: i for i, name in enumerate(header)}

    required = ["FL_DATE", "OP_CARRIER_FL_NUM", "CANCELLED", "CANCELLATION_CODE"]
    missing = [c for c in required if c not in idx]
    if missing:
        raise ValueError(f"Colunas ausentes no {path}: {missing}")

    data_lines = raw.filter(lambda l: l != header_line and not is_header_line(l))
    rows = data_lines.mapPartitions(parse_csv_partition)

    def extract(row: List[str]) -> Tuple[str, str, str, str]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        fl_date = (row[idx["FL_DATE"]] or "").strip()
        fl_num = (row[idx["OP_CARRIER_FL_NUM"]] or "").strip()
        cancelled = (row[idx["CANCELLED"]] or "").strip()
        code = (row[idx["CANCELLATION_CODE"]] or "").strip()
        return fl_date, fl_num, cancelled, code

    return header, rows.map(extract)


def flights_schema() -> StructType:
    return StructType(
        [
            StructField("FL_DATE", StringType(), True),
            StructField("OP_CARRIER", StringType(), True),
            StructField("OP_CARRIER_FL_NUM", IntegerType(), True),
            StructField("ORIGIN", StringType(), True),
            StructField("DEST", StringType(), True),
            StructField("CRS_DEP_TIME", IntegerType(), True),
            StructField("DEP_TIME", DoubleType(), True),
            StructField("DEP_DELAY", DoubleType(), True),
            StructField("TAXI_OUT", DoubleType(), True),
            StructField("WHEELS_OFF", DoubleType(), True),
            StructField("WHEELS_ON", DoubleType(), True),
            StructField("TAXI_IN", DoubleType(), True),
            StructField("CRS_ARR_TIME", IntegerType(), True),
            StructField("ARR_TIME", DoubleType(), True),
            StructField("ARR_DELAY", DoubleType(), True),
            StructField("CANCELLED", DoubleType(), True),
            StructField("CANCELLATION_CODE", StringType(), True),
            StructField("DIVERTED", DoubleType(), True),
            StructField("CRS_ELAPSED_TIME", DoubleType(), True),
            StructField("ACTUAL_ELAPSED_TIME", DoubleType(), True),
            StructField("AIR_TIME", DoubleType(), True),
            StructField("DISTANCE", DoubleType(), True),
            StructField("CARRIER_DELAY", DoubleType(), True),
            StructField("WEATHER_DELAY", DoubleType(), True),
            StructField("NAS_DELAY", DoubleType(), True),
            StructField("SECURITY_DELAY", DoubleType(), True),
            StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True),
            StructField("Unnamed: 27", StringType(), True),
        ]
    )


def read_flights_df(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .schema(flights_schema())
        .load(path)
        .withColumn("FL_DATE_DATE", to_date(col("FL_DATE"), "yyyy-MM-dd"))
    )


def answer_rdd_concepts_text() -> str:
    return (
        "a) RDD é uma coleção distribuída e imutável, particionada no cluster e não otimizada por sí só.\n"
        "b) Importância: paralelismo (partições), tolerância a falhas via lineage, e execução lazy.\n"
        "c) Transformação (map/filter/etc.) produz nova RDD lazy; Ação (count/take/collect/etc.) dispara o job."
    )


def q2_phrase_sample(rdd2009: RDD[Tuple[str, str, str, str]]) -> Tuple[List[str], str]:
    def build_phrase(t: Tuple[str, str, str, str]) -> str:
        fl_date, fl_num, cancelled, _code = t
        was_not = " not" if not cancelled_flag(cancelled) else ""
        return f"\"{fl_num}\" on \"{fl_date}\" was{was_not} cancelled."

    phrases = rdd2009.map(build_phrase)
    head10 = phrases.take(10)
    example = head10[0] if head10 else ""
    return head10, example


def q3_cancelled_sample(rdd2009: RDD[Tuple[str, str, str, str]]) -> List[str]:
    reason_map = {"A": "Airline/Carrier", "B": "Weather", "C": "National Air System", "D": "Security"}

    def to_line(t: Tuple[str, str, str, str]) -> str:
        _fl_date, fl_num, _cancelled, code = t
        code_clean = (code or "").strip() or "N/A"
        meaning = reason_map.get(code_clean, "Unknown/Not provided")
        return f"Flight {fl_num} cancelled due to {code_clean} ({meaning})"

    return rdd2009.filter(lambda t: cancelled_flag(t[2])).map(to_line).take(10)


def q_df_distance_delay_proportions(
    df2011: DataFrame,
) -> Tuple[DataFrame, float, float, Tuple[int, int, int], Tuple[int, int, int]]:
    spark = df2011.sparkSession
    sc = spark.sparkContext
    cores = max(1, sc.defaultParallelism)
    target_partitions = max(16, min(128, cores * 4))

    base = (
        df2011
        .select("CANCELLED", "DISTANCE", "DEP_DELAY", "ARR_DELAY")
        .filter(coalesce(col("CANCELLED"), lit(0.0)) == lit(0.0))
        .filter(col("DISTANCE").isNotNull())
        .coalesce(target_partitions)
    )

    stats = base.agg(
        count(lit(1)).alias("count"),
        mean(col("DISTANCE")).alias("mean"),
        stddev(col("DISTANCE")).alias("stddev"),
        spark_min(col("DISTANCE")).alias("min"),
        spark_max(col("DISTANCE")).alias("max"),
    )

    p33, p66 = base.approxQuantile("DISTANCE", [0.33, 0.66], 0.01)
    if p33 is None or p66 is None:
        raise ValueError("Não foi possível calcular quantis de DISTANCE.")

    delay_any = (coalesce(col("DEP_DELAY"), lit(0.0)) > lit(0.0)) | (coalesce(col("ARR_DELAY"), lit(0.0)) > lit(0.0))

    prox = col("DISTANCE") <= lit(p33)
    med = (col("DISTANCE") > lit(p33)) & (col("DISTANCE") <= lit(p66))
    dist = col("DISTANCE") > lit(p66)

    counts = base.agg(
        spark_sum(when(prox, 1).otherwise(0)).cast("long").alias("total_proximos"),
        spark_sum(when(prox & delay_any, 1).otherwise(0)).cast("long").alias("delayed_proximos"),
        spark_sum(when(med, 1).otherwise(0)).cast("long").alias("total_medio"),
        spark_sum(when(med & delay_any, 1).otherwise(0)).cast("long").alias("delayed_medio"),
        spark_sum(when(dist, 1).otherwise(0)).cast("long").alias("total_distantes"),
        spark_sum(when(dist & delay_any, 1).otherwise(0)).cast("long").alias("delayed_distantes"),
    ).collect()[0]

    def pct(d: int, t: int) -> float:
        return round((d / t) * 100.0, 2) if t else 0.0

    total_prox = int(counts["total_proximos"])
    total_med = int(counts["total_medio"])
    total_dist = int(counts["total_distantes"])
    del_prox = int(counts["delayed_proximos"])
    del_med = int(counts["delayed_medio"])
    del_dist = int(counts["delayed_distantes"])

    rows = [
        ("proximos", total_prox, del_prox, pct(del_prox, total_prox)),
        ("medio", total_med, del_med, pct(del_med, total_med)),
        ("distantes", total_dist, del_dist, pct(del_dist, total_dist)),
    ]

    out_schema = StructType(
        [
            StructField("distance_band", StringType(), False),
            StructField("total_flights", IntegerType(), False),
            StructField("delayed_flights", IntegerType(), False),
            StructField("delay_pct", DoubleType(), False),
        ]
    )

    order_col = when(col("distance_band") == "proximos", lit(1)) \
        .when(col("distance_band") == "medio", lit(2)) \
        .otherwise(lit(3))

    out = spark.createDataFrame(rows, schema=out_schema).orderBy(order_col.asc())
    _ = stats  # mantém a variável 
    return out, float(p33), float(p66), (total_prox, total_med, total_dist), (del_prox, del_med, del_dist)


def q_df_bos_daily_plot(df2011: DataFrame, output_path: str) -> DataFrame:
    bos_daily = (
        df2011
        .select("FL_DATE_DATE", "ORIGIN", "CANCELLED")
        .filter(col("ORIGIN") == lit("BOS"))
        .filter(coalesce(col("CANCELLED"), lit(0.0)) == lit(0.0))
        .filter(col("FL_DATE_DATE").isNotNull())
        .groupBy(col("FL_DATE_DATE"))
        .agg(count(lit(1)).alias("daily_flights"))
        .orderBy(col("FL_DATE_DATE").asc())
    )

    try:
        rows = list(bos_daily.select("FL_DATE_DATE", "daily_flights").toLocalIterator())
        if not rows:
            logger.warning("Sem dados para BOS após filtros. Gráfico não será gerado.")
            return bos_daily

        dates = [r["FL_DATE_DATE"] for r in rows]
        vals = [r["daily_flights"] for r in rows]

        matplotlib.use("Agg")

        plt.figure()
        plt.plot(dates, vals)
        plt.title("Número diário de voos (ORIGIN=BOS) - 2011")
        plt.xlabel("Data")
        plt.ylabel("Voos/dia")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\n[OK] Gráfico salvo em: {output_path}")
    except Exception as e:
        logger.exception("Falha ao gerar gráfico: %s", e)
        print(f"\n[WARN] Não foi possível gerar o gráfico ({e}).")

    return bos_daily


def q_sql_carrier_agg_dual_metrics(spark: SparkSession) -> DataFrame:
    query = """
    SELECT
      OP_CARRIER,
      ROUND(AVG(COALESCE(DEP_DELAY, 0.0) + COALESCE(ARR_DELAY, 0.0)), 2) AS avg_total_delay_raw_min,
      ROUND(
        AVG(
          GREATEST(COALESCE(DEP_DELAY, 0.0), 0.0) +
          GREATEST(COALESCE(ARR_DELAY, 0.0), 0.0)
        ),
        2
      ) AS avg_total_delay_pos_min,
      COUNT(1) AS flights
    FROM flights2011
    WHERE COALESCE(CANCELLED, 0.0) = 0.0
    GROUP BY OP_CARRIER
    """
    return spark.sql(query)


def q_sql_airport_most_weather_delay(spark: SparkSession) -> DataFrame:
    query = """
    SELECT
      DEST AS airport,
      ROUND(SUM(COALESCE(WEATHER_DELAY, 0.0)), 2) AS total_weather_delay_min,
      COUNT(1) AS flights_with_weather_delay
    FROM flights2011
    WHERE COALESCE(CANCELLED, 0.0) = 0.0
      AND COALESCE(WEATHER_DELAY, 0.0) > 0.0
    GROUP BY DEST
    ORDER BY total_weather_delay_min DESC
    LIMIT 10
    """
    return spark.sql(query)


def main() -> None:
    setup_logging()
    report = MarkdownReport(path=OUTPUT_MD_PATH)

    report.h(1, "Relatório - Spark (RDDs, DataFrames, SQL)")
    report.p(f"Gerado em: {datetime.now().isoformat(timespec='seconds')}")
    report.p(
        "Configuração:\n"
        f"- 2009: `{PATH_2009}`\n"
        f"- 2011: `{PATH_2011}`\n"
        f"- Plot: `{OUTPUT_PLOT_PATH}`\n"
        f"- Report: `{OUTPUT_MD_PATH}`\n"
        f"- Master: `{SPARK_MASTER}`\n"
        f"- Spark log level: `{SPARK_LOG_LEVEL}`\n"
        f"- spark.sql.debug.maxToStringFields: `{DEFAULT_SQL_DEBUG_MAX_TO_STRING_FIELDS}`"
    )

    banner("Configuração usada (sem CLI)")
    print(f"PATH_2009         = {PATH_2009}")
    print(f"PATH_2011         = {PATH_2011}")
    print(f"OUTPUT_PLOT_PATH  = {OUTPUT_PLOT_PATH}")
    print(f"OUTPUT_MD_PATH    = {OUTPUT_MD_PATH}")
    print(f"SPARK_MASTER      = {SPARK_MASTER}")
    print(f"LOG_LEVEL         = {LOG_LEVEL}")
    print(f"SPARK_LOG_LEVEL   = {SPARK_LOG_LEVEL}")
    print(f"maxToStringFields = {DEFAULT_SQL_DEBUG_MAX_TO_STRING_FIELDS}")

    if not os.path.exists(PATH_2009):
        raise FileNotFoundError(f"Arquivo 2009.csv não encontrado em: {PATH_2009}")
    if not os.path.exists(PATH_2011):
        raise FileNotFoundError(f"Arquivo 2011.csv não encontrado em: {PATH_2011}")

    spark = build_spark()
    sc = spark.sparkContext

    try:
        report.h(2, "Spark RDDs")

        report.h(3, "Q1) O que é RDD? Importância? Transformação vs Ação?")
        q1_text = answer_rdd_concepts_text()
        banner("Spark RDDs - Q1 (conceitos)")
        print(q1_text)
        report.p(q1_text)

        header2009, rdd2009 = read_2009_slim_rdd(sc, PATH_2009)
        rdd2009.persist(StorageLevel.DISK_ONLY)

        report.h(3, "RDD 2009 (amostra)")
        report.code(str(header2009), language="text")

        rdd2009_head = rdd2009.take(10)
        rdd_head_txt = print_rdd_head("RDD 2009 (tuplas: FL_DATE, FL_NUM, CANCELLED, CODE)", rdd2009_head, 10)
        report.code(rdd_head_txt, language="text")

        report.h(3, "Q2) Frase: \"OP_CARRIER_FL_NUM\" on \"FL_DATE\" was/was not cancelled")
        q2_head, q2_example = q2_phrase_sample(rdd2009)
        q2_txt = print_rdd_head("RDD Q2 - Frases (amostra)", q2_head, 10)
        banner("RDDs - Q2 (variável/string exemplo)")
        print(q2_example)
        report.p(f"Exemplo de variável (string): `{q2_example}`")
        report.code(q2_txt, language="text")

        report.h(3, "Q3) Cancelados: \"Flight NUMBER cancelled due to CODE\"")
        q3_head = q3_cancelled_sample(rdd2009)
        q3_txt = print_rdd_head("RDD Q3 - Cancelados (amostra)", q3_head, 10)
        report.p("Dicionário: A=Airline/Carrier, B=Weather, C=National Air System, D=Security.")
        report.code(q3_txt, language="text")

        report.h(2, "Spark DataFrame (2011.csv)")
        df2011 = read_flights_df(spark, PATH_2011)

        report.h(3, "Amostra do DataFrame (primeiras 10 linhas)")
        df_show = show_df_head("DataFrame 2011 (raw)", df2011, n=10)
        report.code(df_show, language="text")

        report.h(3, "Q1) Faixas de distância e proporção (%) de voos com atraso")
        agg, p33, p66, totals, delayed = q_df_distance_delay_proportions(df2011)

        banner("DataFrame - Q1 (cortes escolhidos para DISTANCE)")
        print(f"Corte 1 (≈33%): {p33:.2f}")
        print(f"Corte 2 (≈66%): {p66:.2f}")
        print("Faixas: proximos <= p33; medio (p33, p66]; distantes > p66")

        report.p(
            f"Cortes (aprox.): p33={p33:.2f}, p66={p66:.2f}\n\n"
            "Definição de atraso usada: `(DEP_DELAY > 0) OR (ARR_DELAY > 0)` (cancelados excluídos)."
        )
        report.p(f"Verificação: totais={totals} (soma={sum(totals)}), atrasados={delayed} (soma={sum(delayed)})")

        agg_show = show_df_head("DataFrame Q1 - Resultado por faixa", agg, n=10)
        report.code(agg_show, language="text")

        report.h(3, "Q2) Visualização: número diário de voos com origem BOS")
        bos_daily = q_df_bos_daily_plot(df2011, OUTPUT_PLOT_PATH)
        bos_show = show_df_head("DataFrame Q2 - BOS diário", bos_daily, n=10)
        report.p(f"Gráfico salvo em: `{OUTPUT_PLOT_PATH}`")
        report.p(f"![BOS daily flights]({os.path.relpath(OUTPUT_PLOT_PATH, os.path.dirname(os.path.abspath(OUTPUT_MD_PATH)))})")
        report.code(bos_show, language="text")

        report.h(2, "Spark SQL")

        df_sql = df2011.select(
            "OP_CARRIER", "DEP_DELAY", "ARR_DELAY", "CANCELLED",
            "WEATHER_DELAY", "DEST"
        )
        df_sql.createOrReplaceTempView("flights2011")

        report.h(3, "Q6) Operadoras mais pontuais em média (atraso total saída+chegada)")
        report.p(
            "Métricas:\n"
            "- `avg_total_delay_raw_min`: média de `DEP_DELAY + ARR_DELAY` (pode ser negativa)\n"
            "- `avg_total_delay_pos_min`: média de `max(DEP_DELAY,0) + max(ARR_DELAY,0)` (apenas tardança)\n"
        )

        carrier_agg = q_sql_carrier_agg_dual_metrics(spark).cache()

        punctual_by_pos = (
            carrier_agg
            .orderBy(col("avg_total_delay_pos_min").asc(), col("avg_total_delay_raw_min").asc(), col("flights").desc())
            .limit(10)
        )
        punctual_by_raw = (
            carrier_agg
            .orderBy(col("avg_total_delay_raw_min").asc(), col("avg_total_delay_pos_min").asc(), col("flights").desc())
            .limit(10)
        )

        pun_pos_show = show_df_head("Spark SQL Q6 - Ranking por avg_total_delay_pos_min (top 10)", punctual_by_pos, n=10)
        report.h(4, "Ranking por tardança (avg_total_delay_pos_min)")
        report.code(pun_pos_show, language="text")

        pun_raw_show = show_df_head("Spark SQL Q6 - Ranking por avg_total_delay_raw_min (top 10)", punctual_by_raw, n=10)
        report.h(4, "Ranking por atraso 'bruto' (avg_total_delay_raw_min)")
        report.code(pun_raw_show, language="text")

        report.h(3, "Q7) Aeroporto com mais atrasos por questões de clima")
        weather = q_sql_airport_most_weather_delay(spark)
        wea_show = show_df_head("Spark SQL Q7 - Clima (DEST, top 10)", weather, n=10)
        report.code(wea_show, language="text")

        top1 = weather.take(1)
        answer_q7 = (
            f"Aeroporto (DEST) com maior atraso total por clima: {top1[0]['airport']} "
            f"com {top1[0]['total_weather_delay_min']} min"
        ) if top1 else "Não há voos com WEATHER_DELAY > 0 após filtros."
        banner("Resposta direta (SQL Q7 - top 1 por DEST)")
        print(answer_q7)
        report.p(f"Resposta direta: **{answer_q7}**")

    finally:
        try:
            report.save()
            banner("Relatório salvo")
            print(f"[OK] Markdown: {OUTPUT_MD_PATH}")
        finally:
            spark.stop()


if __name__ == "__main__":
    main()






