import sys
from pathlib import Path

import duckdb
import pandas as pd

# csv 
FILE_PATH = "/Users/akatsurada/Documents/INSPER/BigData/Aula6/Lab2/healthcare-dataset-stroke-data.csv"

# Utils
def escapar_aspas_simples(texto: str) -> str:
    return texto.replace("'", "''")


def garantir_arquivo(caminho: str) -> Path:
    p = Path(caminho)
    if not p.is_file():
        sys.stderr.write(
            f"Arquivo nao encontrado no caminho especificado  {p.resolve()}\n"
        )
        sys.exit(1)
    return p

# Inicia conn
def iniciar_conexao() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    return con



# Cria view p/ cleaning e casting
def criar_view_bruta(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    caminho_escapado = escapar_aspas_simples(csv_path.as_posix())
    sql = f"""
        CREATE OR REPLACE VIEW raw_csv AS
        SELECT *
        FROM read_csv_auto(
            '{caminho_escapado}',
            header = true,
            all_varchar = true,
            normalize_names = true
        );
    """
    con.execute(sql)

# Cols esperadas
def validar_colunas_esperadas(con: duckdb.DuckDBPyConnection) -> None:
    esperadas = {
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    }
    linhas = con.execute(
        """
        SELECT lower(column_name)
        FROM information_schema.columns
        WHERE table_name = 'raw_csv'
        """
    ).fetchall()
    existentes = {linha[0] for linha in linhas}
    faltantes = sorted(esperadas - existentes)
    if faltantes:
        msg = "Colunas ausentes na origem  " + ", ".join(faltantes) + "\n"
        sys.stderr.write(msg)
        sys.exit(1)

# Tabela + nulls + espaços
def criar_tabela_pacientes(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE patients AS
        SELECT
            TRY_CAST(id AS BIGINT) AS id,
            CASE
                WHEN gender IS NULL OR length(trim(gender)) = 0 THEN NULL
                ELSE trim(gender)
            END AS gender,
            TRY_CAST(age AS DOUBLE) AS age,
            TRY_CAST(hypertension AS INTEGER) AS hypertension,
            TRY_CAST(heart_disease AS INTEGER) AS heart_disease,
            CASE
                WHEN ever_married IS NULL OR length(trim(ever_married)) = 0 THEN NULL
                ELSE trim(ever_married)
            END AS ever_married,
            CASE
                WHEN work_type IS NULL OR length(trim(work_type)) = 0 THEN NULL
                ELSE trim(work_type)
            END AS work_type,
            CASE
                WHEN residence_type IS NULL OR length(trim(residence_type)) = 0 THEN NULL
                ELSE trim(residence_type)
            END AS residence_type,
            TRY_CAST(avg_glucose_level AS DOUBLE) AS avg_glucose_level,
            CASE
                WHEN bmi IS NULL THEN NULL
                WHEN lower(trim(bmi)) IN ('n/a','na','nan','null','') THEN NULL
                ELSE TRY_CAST(bmi AS DOUBLE)
            END AS bmi,
            CASE
                WHEN smoking_status IS NULL THEN NULL
                WHEN lower(trim(smoking_status)) IN ('n/a','na','nan','unknown','') THEN NULL
                ELSE trim(smoking_status)
            END AS smoking_status,
            TRY_CAST(stroke AS INTEGER) AS stroke
        FROM raw_csv
        """
    )
    # Analyze das estatisticas 
    con.execute("ANALYZE patients")

# Pessoas c/ avc por tipo de trabalho, ordenacao decrescente e alias 
def consulta_avc_por_trabalho(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
        SELECT
            COALESCE(work_type, 'desconhecido') AS work_type,
            COUNT(*) AS work_type_count
        FROM patients
        WHERE stroke = 1
        GROUP BY 1
        ORDER BY work_type_count DESC, work_type ASC
    """
    return con.execute(sql).fetchdf()

# Total de pacientes por genero e percentual em relacao ao total
def consulta_total_por_genero_percentual(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
        WITH base AS (
            SELECT
                COALESCE(gender, 'desconhecido') AS gender,
                COUNT(*) AS total_por_genero
            FROM patients
            GROUP BY COALESCE(gender, 'desconhecido')
        ),
        total AS (
            SELECT SUM(total_por_genero) AS total_pacientes
            FROM base
        )
        SELECT
            base.gender,
            base.total_por_genero,
            ROUND(
                100.0 * base.total_por_genero::DOUBLE / total.total_pacientes,
                2
            ) AS percentual_base
        FROM base, total
        ORDER BY base.total_por_genero DESC, base.gender ASC
    """
    return con.execute(sql).fetchdf()


# Pacientes do sexo masculino com avc e percentual sobre o total 
def consulta_masculinos_com_avc_percentual(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
        WITH total AS (
            SELECT COUNT(*) AS total_pacientes
            FROM patients
        ),
        masculino_avc AS (
            SELECT COUNT(*) AS total_masculino_com_avc
            FROM patients
            WHERE lower(gender) = 'male'
              AND stroke = 1
        )
        SELECT
            masculino_avc.total_masculino_com_avc,
            ROUND(
                100.0 * masculino_avc.total_masculino_com_avc::DOUBLE
                / total.total_pacientes,
                2
            ) AS percentual_base
        FROM masculino_avc, total
    """
    return con.execute(sql).fetchdf()


# Total de pacientes com avc por faixa etaria em bandas de dez anos e faixa agregada
def consulta_avc_por_faixa_etaria(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
        WITH base AS (
            SELECT
                CASE
                    WHEN age IS NULL THEN NULL
                    WHEN age >= 80 THEN '80 ou mais'
                    ELSE CAST(10 * FLOOR(age / 10) AS INTEGER) || ' a ' ||
                         CAST(10 * FLOOR(age / 10) + 9 AS INTEGER)
                END AS faixa_etaria,
                CASE
                    WHEN age IS NULL THEN NULL
                    WHEN age >= 80 THEN 80
                    ELSE CAST(10 * FLOOR(age / 10) AS INTEGER)
                END AS faixa_inicio
            FROM patients
            WHERE stroke = 1
        )
        SELECT
            faixa_etaria,
            COUNT(*) AS total_pacientes
        FROM base
        WHERE faixa_etaria IS NOT NULL
        GROUP BY faixa_etaria, faixa_inicio
        ORDER BY faixa_inicio
    """
    return con.execute(sql).fetchdf()

# Print
def imprimir_secao(titulo: str, df: pd.DataFrame) -> None:
    print("\n" + titulo)
    if df.empty:
        print("Sem registros")
        return
    print(df.to_string(index=False))


def main() -> None:
    caminho = garantir_arquivo(FILE_PATH)
    con = iniciar_conexao()
    print()

    criar_view_bruta(con, caminho)
    validar_colunas_esperadas(con)
    criar_tabela_pacientes(con)

    # Consultas solicitadas
    df_q1 = consulta_avc_por_trabalho(con)
    df_q2 = consulta_total_por_genero_percentual(con)
    df_q3 = consulta_masculinos_com_avc_percentual(con)
    df_q4 = consulta_avc_por_faixa_etaria(con)

    # Print
    imprimir_secao(
        "Consulta 1 numero de pessoas com avc por tipo de trabalho ordenado e com coluna work_type_count",
        df_q1,
    )
    imprimir_secao(
        "Consulta 2 total de pacientes por genero e percentual sobre a base",
        df_q2,
    )
    imprimir_secao(
        "Consulta 3 total de pacientes do sexo masculino com avc e percentual sobre a base",
        df_q3,
    )
    imprimir_secao(
        "Consulta 4 total de pacientes com avc por faixa etaria",
        df_q4,
    )

if __name__ == "__main__":
    main()
