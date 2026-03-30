#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dados de patentes c/ PySpark usando RDD, DataFrame e SparkSQL.
- RDD: take → remover cabeçalho → filter → map → reduceByKey → takeOrdered
- DF / SQL: alternativas modernas/eficientes.
- Strategy/Registry Pattern para engine/sink/format (sem if/elif/else “crescentes”).
- Chamado via run({...}) com defaults (sem CLI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace, fields
from operator import add
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import csv
from pyspark import StorageLevel
from pyspark.rdd import RDD
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T


# -------------------------
# Config + Defaults
# -------------------------
@dataclass(frozen=True)
class Config:
    APP_NAME: str = "TopCitedPatents"
    INPUT_PATH: str = "citations.csv"        # default, pode ser sobrescrito em run({...})
    TOP_N: int = 10
    MIN_PARTITIONS: Optional[int] = None
    SHUFFLE_PARTITIONS: Optional[int] = None
    INSPECT: bool = False
    ENGINE: str = "rdd"                      # "rdd" | "df" | "sql"
    OUTPUT_PATH: Optional[str] = None        # None → stdout
    OUTPUT_FORMAT: str = "parquet"           # "csv" | "parquet" (se OUTPUT_PATH definido)
    COALESCE_N: int = 1
    VERBOSITY: int = 0                       # 0 WARN | 1 INFO | 2 DEBUG

DEFAULTS = Config()


def with_overrides(base: Config, **overrides) -> Config:
    """
    Aplica sobreposições válidas (ignora chaves desconhecidas ou valores None).
    """
    valid = {f.name for f in fields(Config)}
    clean = {k: v for k, v in overrides.items() if (k in valid and v is not None)}
    return replace(base, **clean)


# -------------------------
# Infra: Spark + Logging
# -------------------------
def build_spark(app_name: str, shuffle_partitions: Optional[int]) -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.caseSensitive", "false")
    )
    # aplicar config opcional sem if/elif
    extras = {} if shuffle_partitions is None else {"spark.sql.shuffle.partitions": str(shuffle_partitions)}
    for k, v in extras.items():
        builder = builder.config(k, v)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


# -------------------------
# Utilitários de parsing (RDD)
# -------------------------
def is_header_line(line: str) -> bool:
    """
    Detecta cabeçalho 'CITING,CITED' (tolerante a BOM/aspas/espaços).
    """
    s = line.lstrip("\ufeff").strip().replace('"', "")
    if not s:
        return False
    if s.upper().startswith("CITING,CITED"):
        return True
    try:
        row = next(csv.reader([s]))
        if len(row) >= 2:
            return row[0].strip().upper() == "CITING" and row[1].strip().upper() == "CITED"
    except Exception:
        pass
    return False


def parse_citation_partition(lines: Iterable[str]) -> Iterator[Tuple[int, int]]:
    """
    Parser por partição (csv.reader) — robusto e rápido.
    Emite (citing:int, cited:int).
    """
    cleaned_iter = (ln.lstrip("\ufeff").strip() for ln in lines)
    reader = csv.reader(cleaned_iter)
    for row in reader:
        if not row:
            continue
        try:
            c0 = row[0].strip().replace('"', '')
            c1 = row[1].strip().replace('"', '')
        except IndexError:
            continue
        if c0.upper() == "CITING" and c1.upper() == "CITED":
            continue
        try:
            citing = int(c0)
            cited = int(c1)
        except ValueError:
            continue
        yield citing, cited


def resolve_probe(inspect: bool, logger: logging.Logger) -> Callable[[RDD[str]], None]:
    """
    Registry indexado por bool: True→loga take(5), False→no-op.
    Evita if/else no caminho crítico.
    """
    return {
        False: (lambda rdd: None),
        True:  (lambda rdd: logger.info("Amostra (5): %s", rdd.take(5))),
    }[inspect]


# -------------------------
# Leitura DF padronizada
# -------------------------
def read_citations_df(spark: SparkSession, input_path: str) -> DataFrame:
    schema = T.StructType([
        T.StructField("CITING", T.LongType(), True),
        T.StructField("CITED",  T.LongType(), True),
    ])
    return (
        spark.read
        .option("header", True)
        .option("mode", "DROPMALFORMED")
        .schema(schema)
        .csv(input_path)
        .select("CITING", "CITED")
        .where(F.col("CITED").isNotNull())
    )


# -------------------------
# Engines (todos retornam DF: [CITED, citation_count])
# -------------------------
def run_engine_rdd(spark: SparkSession, cfg: Config) -> DataFrame:
    """
    RDD → passos explícitos requeridos → Top-N → DataFrame unificado.
    """
    logger = logging.getLogger("engine-rdd")
    sc = spark.sparkContext

    raw_lines: RDD[str] = sc.textFile(cfg.INPUT_PATH, minPartitions=cfg.MIN_PARTITIONS or None)
    resolve_probe(cfg.INSPECT, logger)(raw_lines)  # primeira ação (take) opcional

    no_header: RDD[str] = raw_lines.filter(lambda ln: not is_header_line(ln))
    filtered: RDD[str] = no_header.filter(lambda ln: ln and "," in ln)

    edges: RDD[Tuple[int, int]] = filtered.mapPartitions(parse_citation_partition)
    edges.persist(StorageLevel.MEMORY_AND_DISK)

    cited_one: RDD[Tuple[int, int]] = edges.map(lambda pair: (pair[1], 1))
    counts_by_cited: RDD[Tuple[int, int]] = cited_one.reduceByKey(add)

    top_k: List[Tuple[int, int]] = counts_by_cited.takeOrdered(cfg.TOP_N, key=lambda kv: (-kv[1], kv[0]))
    return spark.createDataFrame(top_k, schema=["CITED", "citation_count"])


def run_engine_df(spark: SparkSession, cfg: Config) -> DataFrame:
    df = read_citations_df(spark, cfg.INPUT_PATH)
    return (
        df.groupBy("CITED")
          .count()
          .withColumnRenamed("count", "citation_count")
          .orderBy(F.desc("citation_count"), F.asc("CITED"))
          .limit(cfg.TOP_N)
    )


def run_engine_sql(spark: SparkSession, cfg: Config) -> DataFrame:
    df = read_citations_df(spark, cfg.INPUT_PATH)
    df.createOrReplaceTempView("citations")
    query = f"""
        SELECT CITED, COUNT(*) AS citation_count
        FROM citations
        GROUP BY CITED
        ORDER BY citation_count DESC, CITED ASC
        LIMIT {int(cfg.TOP_N)}
    """
    return spark.sql(query)


# -------------------------
# Resolvers (registries)
# -------------------------
def resolve_engine(engine_name: str) -> Callable[[SparkSession, Config], DataFrame]:
    name = (engine_name or "").strip().lower()
    registry: Dict[str, Callable[[SparkSession, Config], DataFrame]] = {
        "rdd": run_engine_rdd,
        "df":  run_engine_df,
        "sql": run_engine_sql,
    }
    try:
        return registry[name]
    except KeyError as e:
        raise ValueError(f"Engine desconhecido: {engine_name!r}. Use 'rdd' | 'df' | 'sql'.") from e


def make_fs_writer(output_path: str, fmt: str, coalesce_n: int) -> Callable[[SparkSession, DataFrame], None]:
    def _writer(_spark: SparkSession, df: DataFrame) -> None:
        out_df = df.coalesce(coalesce_n)
        fmt_key = (fmt or "").strip().lower()
        format_registry: Dict[str, Callable[[DataFrame], None]] = {
            "csv":     lambda d: d.write.mode("overwrite").option("header", True).csv(output_path),
            "parquet": lambda d: d.write.mode("overwrite").parquet(output_path),
        }
        try:
            format_registry[fmt_key](out_df)
        except KeyError as e:
            raise ValueError(f"Formato desconhecido: {fmt!r}. Use 'csv' ou 'parquet'.") from e
    return _writer


def stdout_writer() -> Callable[[SparkSession, DataFrame], None]:
    def _writer(_spark: SparkSession, df: DataFrame) -> None:
        rows = df.collect()
        print("CITED,citation_count")
        for r in rows:
            print(f"{int(r['CITED'])},{int(r['citation_count'])}")
    return _writer


def resolve_writer(output_path: Optional[str], fmt: str, coalesce_n: int) -> Callable[[SparkSession, DataFrame], None]:
    # registry indexado por bool: False→stdout, True→filesystem
    sink_registry: Dict[bool, Callable[[SparkSession, DataFrame], None]] = {
        False: stdout_writer(),
        True:  make_fs_writer(output_path, fmt, coalesce_n),
    }
    return sink_registry[bool(output_path)]


# -------------------------
# Public API (no CLI)
# -------------------------
def run(overrides: Optional[dict] = None) -> None:
    """
    Ponto único de execução.
    Ex.: run({"ENGINE":"df","INPUT_PATH":"data/citations.csv","OUTPUT_PATH":"out/top10","OUTPUT_FORMAT":"parquet"})
    """
    cfg = with_overrides(DEFAULTS, **(overrides or {}))
    setup_logging(cfg.VERBOSITY)
    spark = build_spark(cfg.APP_NAME, cfg.SHUFFLE_PARTITIONS)
    try:
        engine_runner = resolve_engine(cfg.ENGINE)
        writer = resolve_writer(cfg.OUTPUT_PATH, cfg.OUTPUT_FORMAT, cfg.COALESCE_N)
        result_df = engine_runner(spark, cfg)
        writer(spark, result_df)
        logging.getLogger("main").info("Finalizado com sucesso.")
    finally:
        spark.stop()


# Execução direta (útil para `python top_cited_patents_strategies.py`)
if __name__ == "__main__":
    run()  # usa defaults
