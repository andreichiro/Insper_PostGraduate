# nyc_taxi_eda_ps_min.py
# -*- coding: utf-8 -*-

# bloco inicial exigido sem alteracao
import seaborn as sns
import pyspark.sql.functions as f
from pyspark.sql.types import StringType
from matplotlib import pyplot as plt
from pyspark.sql import SparkSession
spark = SparkSession \
                    .builder \
                    .master('local[*]') \
                    .appName('nyctaxi_andre') \
                    .getOrCreate()
# fim do bloco fixo

# ajustes simples e imports adicionais
spark.conf.set("spark.sql.ansi.enabled", "false")  # corrige erro do pandas on spark em ansi mode

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark.pandas as ps  # pandas on spark

from pyspark.sql.types import (
    StructType, StructField,
    TimestampType, FloatType, IntegerType
)

plt.switch_backend("Agg")
sns.set()

# caminhos e pastas
INPUT_PATH = "/Users/akatsurada/Documents/INSPER/BigData/Aula8/train.csv"
OUTDIR = Path("./outputs_nyc_taxi_min")
OUTDIR.mkdir(parents=True, exist_ok=True)

# schema fornecido anteriormente
labels = (
    ('key', TimestampType()),
    ('fare_amount', FloatType()),
    ('pickup_datetime', TimestampType()),
    ('pickup_longitude', FloatType()),
    ('pickup_latitude', FloatType()),
    ('dropoff_longitude', FloatType()),
    ('dropoff_latitude', FloatType()),
    ('passenger_count', IntegerType())
)
schema = StructType([StructField(x[0], x[1], True) for x in labels])

# somente colunas permitidas
ALLOWED = [
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "rate_code",
    "store_and_fwd_flag",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "imp_surcharge",
    "total_amount",
    "pickup_location_id",
    "dropoff_location_id"
]

# leitura simples com schema fixo
print("Lendo dados")
df = spark.read.csv(INPUT_PATH, header=True, schema=schema)

# assegura uso somente de colunas permitidas que existirem
present_allowed = [c for c in ["pickup_datetime", "passenger_count", "fare_amount"] if c in df.columns]
df = df.select(*present_allowed)

# amostra de um por cento como no trecho dado
df = df.sample(withReplacement=False, fraction=0.01, seed=42)
df.cache()
print(f"Linhas apos amostra {df.count()}")

# exploracao 1 frequencia mensal
df_date_quality = df.withColumn('month', f.date_trunc("month", f.col('pickup_datetime')))
sdf_month = df_date_quality.groupby('month').count().orderBy('month')
ps_month = sdf_month.pandas_api()
pdf_month = ps_month.to_pandas().set_index("month")["count"]
ax = pdf_month.plot(kind="line", figsize=(15, 5), title="Frequencia mensal de corridas")
ax.set_xlabel("mes")
ax.set_ylabel("quantidade")
plt.tight_layout()
plt.savefig(str(OUTDIR / "freq_mensal.png"))
plt.close()
ps_month.to_pandas().to_csv(OUTDIR / "freq_mensal.csv", index=False)
print("Grafico frequencia mensal salvo")

# limpeza simples baseada 
df2 = df.filter(
    (f.col('fare_amount') < 10000) &
    (f.col('fare_amount') > 0) &
    (f.col('passenger_count') > 5)
)
df.unpersist()
df2.cache()
print(f"Linhas apos filtro basico {df2.count()}")

# exploracao 2 frequencia diaria
df_daily_trips = df2.withColumn('daily', f.date_trunc("day", f.col('pickup_datetime')))
sdf_daily = df_daily_trips.groupBy('daily').count().orderBy('daily')
ps_daily = sdf_daily.pandas_api()
pdf_daily = ps_daily.to_pandas().set_index("daily")["count"]
ax = pdf_daily.plot(kind="line", figsize=(15, 5), title="Frequencia diaria de corridas")
ax.set_xlabel("dia")
ax.set_ylabel("quantidade")
plt.tight_layout()
plt.savefig(str(OUTDIR / "freq_diaria.png"))
plt.close()
ps_daily.to_pandas().to_csv(OUTDIR / "freq_diaria.csv", index=False)
print("Grafico frequencia diaria salvo")

# exploracao 3 variacao por hora
df_hours_trips = df2.withColumn("month", f.date_trunc("month", f.col('pickup_datetime')))
df_hours_trips = df_hours_trips.withColumn("hour", f.hour(f.col('pickup_datetime')))
sdf_hours = df_hours_trips.groupBy(["month", "hour"]).count()
# agregacao por hora para uma linha simples 
sdf_hours_by_hour = sdf_hours.groupBy("hour").sum("count").withColumnRenamed("sum(count)", "count")
ps_hours = sdf_hours_by_hour.orderBy("hour").pandas_api()
pdf_hours = ps_hours.to_pandas().set_index("hour")["count"]
ax = pdf_hours.plot(kind="line", figsize=(15, 5), title="Variacao de corridas por hora")
ax.set_xlabel("hora")
ax.set_ylabel("quantidade")
plt.tight_layout()
plt.savefig(str(OUTDIR / "freq_por_hora.png"))
plt.close()
sdf_hours.orderBy("month", "hour").pandas_api().to_pandas().to_csv(OUTDIR / "freq_por_hora_detalhe.csv", index=False)
print("Grafico frequencia por hora salvo")

# correlacoes numericas apenas com colunas permitidas presentes
num_cols = [c for c in ["fare_amount", "trip_distance", "passenger_count", "extra", "mta_tax", "tip_amount", "tolls_amount", "imp_surcharge", "total_amount"] if c in df2.columns]
if "fare_amount" in num_cols and len(num_cols) > 1:
    # matriz simples com stat.corr par a par
    mat = []
    for c1 in num_cols:
        row = []
        for c2 in num_cols:
            if c1 == c2:
                row.append(1.0)
            else:
                try:
                    row.append(float(df2.stat.corr(c1, c2)))
                except Exception:
                    row.append(np.nan)
        mat.append(row)
    corr_pdf = pd.DataFrame(mat, index=num_cols, columns=num_cols)
    corr_pdf.to_csv(OUTDIR / "correlacoes.csv", index=True)
    # barra com correlacao contra fare_amount usando pandas on spark
    fare_corr = corr_pdf["fare_amount"].drop(labels=["fare_amount"])
    corr_bar_pdf = fare_corr.abs().sort_values(ascending=False).to_frame("abs_corr")
    if len(corr_bar_pdf) > 0:
        ax = corr_bar_pdf.plot(kind="bar", figsize=(10, 4), title="Correlacao absoluta com fare_amount")
        ax.set_xlabel("variavel")
        ax.set_ylabel("correlacao")
        plt.tight_layout()
        plt.savefig(str(OUTDIR / "corr_com_fare_amount.png"))
        plt.close()
        top_var = fare_corr.abs().idxmax()
        top_val = float(fare_corr.loc[top_var])
        with open(OUTDIR / "top_correlacao.txt", "w", encoding="utf-8") as fh:
            fh.write(f"Mais correlacionada com fare_amount é {top_var} com {top_val:.6f}\n")
        print(f"Mais correlacionada com fare_amount é {top_var} com {top_val:.6f}")
else:
    print("Colunas numericas insuficientes para correlacao")
    with open(OUTDIR / "correlacoes.txt", "w", encoding="utf-8") as fh:
        fh.write("Colunas numericas insuficientes para correlacao\n")

# salva um parquet com o df2 para consulta
df2.write.mode("overwrite").parquet(str(OUTDIR / "nyc_taxi_clean.parquet"))
print(f"Saidas salvas em {OUTDIR.resolve()}")
