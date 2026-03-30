from __future__ import annotations

from pathlib import Path
import sys
import csv
import duckdb
import pandas as pd


# Caminhos, parâmetros e arquivos necessários
DATA_DIR = Path(__file__).resolve().parent
DB_FILE = DATA_DIR / "movies.duckdb"
EXPORT_DIR = DATA_DIR / "export"
EXPORT_FORMAT = "parquet"  # aceitar "parquet" ou "csv"
TOPN = 25
OVERWRITE = True

REQUIRED_FILES = [
    "actors.dat",
    "genres.dat",
    "movies.dat",
    "tag_names.dat",
    "tags.dat",
]


# Verificação de existência dos arquivos de entrada
def garantir_entradas(data_dir: Path) -> None:
    faltando = [n for n in REQUIRED_FILES if not (data_dir / n).exists()]
    if faltando:
        print(f"Erro: arquivos ausentes em {data_dir}: {faltando}", file=sys.stderr)
        sys.exit(2)


# Leitura robusta de arquivos TSV
def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=names,
            dtype=str,
            keep_default_na=False,
            na_values=[],
            engine="c",
        )
    except Exception:
        return pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=names,
            dtype=str,
            keep_default_na=False,
            na_values=[],
            engine="python",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )


# Padronização de strings
def _padronizar_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# Troca de marcadores de ausentes
def _trocar_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(r"\N", pd.NA, regex=False)
    return df


# Carregamento, limpeza e tipagem das tabelas
def carregar_dados(data_dir: Path):
    movies = _read_dat(data_dir / "movies.dat", ["mid", "title", "year", "rating", "num_ratings"])
    actors = _read_dat(data_dir / "actors.dat", ["mid", "name", "cast_position"])
    genres = _read_dat(data_dir / "genres.dat", ["mid", "genre"])
    tags = _read_dat(data_dir / "tags.dat", ["mid", "tid"])
    tag_names = _read_dat(data_dir / "tag_names.dat", ["tid", "tag"])

    movies = _padronizar_strings(movies)
    actors = _padronizar_strings(actors)
    genres = _padronizar_strings(genres)
    tags = _padronizar_strings(tags)
    tag_names = _padronizar_strings(tag_names)

    movies = _trocar_missing(movies)
    actors = _trocar_missing(actors)
    genres = _trocar_missing(genres)
    tags = _trocar_missing(tags)
    tag_names = _trocar_missing(tag_names)

    movies["mid"] = pd.to_numeric(movies["mid"], errors="coerce").astype("Int64")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").astype("Int64")
    movies["rating"] = pd.to_numeric(movies["rating"], errors="coerce").astype("float32")
    movies["num_ratings"] = pd.to_numeric(movies["num_ratings"], errors="coerce").astype("Int64")

    actors["mid"] = pd.to_numeric(actors["mid"], errors="coerce").astype("Int64")
    actors["cast_position"] = pd.to_numeric(actors["cast_position"], errors="coerce").astype("Int64")

    genres["mid"] = pd.to_numeric(genres["mid"], errors="coerce").astype("Int64")

    tags["mid"] = pd.to_numeric(tags["mid"], errors="coerce").astype("Int64")
    tags["tid"] = pd.to_numeric(tags["tid"], errors="coerce").astype("Int64")

    tag_names["tid"] = pd.to_numeric(tag_names["tid"], errors="coerce").astype("Int64")

    return movies, actors, genres, tags, tag_names


# Conexão persistente com DuckDB
def conectar_persistente(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))


# Materialização das tabelas e criação de visões
def materializar(
    con: duckdb.DuckDBPyConnection,
    movies: pd.DataFrame,
    actors: pd.DataFrame,
    genres: pd.DataFrame,
    tags: pd.DataFrame,
    tag_names: pd.DataFrame,
) -> None:
    con.register("movies_df", movies)
    con.register("actors_df", actors)
    con.register("genres_df", genres)
    con.register("tags_df", tags)
    con.register("tag_names_df", tag_names)

    con.sql("CREATE OR REPLACE TABLE movies AS SELECT * FROM movies_df")
    con.sql("CREATE OR REPLACE TABLE actors AS SELECT * FROM actors_df")
    con.sql("CREATE OR REPLACE TABLE genres AS SELECT * FROM genres_df")
    con.sql("CREATE OR REPLACE TABLE tags AS SELECT * FROM tags_df")
    con.sql("CREATE OR REPLACE TABLE tag_names AS SELECT * FROM tag_names_df")

    con.sql(
        """
        CREATE OR REPLACE VIEW v_cast AS
        SELECT m.mid, m.title, m.year, a.name, a.cast_position
        FROM movies m
        JOIN actors a USING (mid)
        """
    )
    con.sql(
        """
        CREATE OR REPLACE VIEW v_genre_ratings AS
        SELECT g.genre, m.mid, m.rating, m.title, m.year
        FROM genres g
        JOIN movies m USING (mid)
        """
    )
    con.sql(
        """
        CREATE OR REPLACE VIEW v_tags AS
        SELECT t.mid, t.tid, tn.tag
        FROM tags t
        JOIN tag_names tn USING (tid)
        """
    )


# Garantia de pasta de saída
def garantir_pasta(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Exportação de tabela completa
def exportar_tabela(
    con: duckdb.DuckDBPyConnection, tabela: str, pasta: Path, formato: str, sobrescrever: bool
) -> Path:
    garantir_pasta(pasta)
    if formato == "parquet":
        out = pasta / f"{tabela}.parquet"
        if out.exists() and not sobrescrever:
            return out
        con.sql(f"COPY (SELECT * FROM {tabela}) TO '{out.as_posix()}' (FORMAT PARQUET)")
        return out
    if formato == "csv":
        out = pasta / f"{tabela}.csv"
        if out.exists() and not sobrescrever:
            return out
        con.sql(f"COPY (SELECT * FROM {tabela}) TO '{out.as_posix()}' (HEADER, DELIMITER ',')")
        return out
    raise ValueError("formato deve ser parquet ou csv")


# Exportação de consulta
def exportar_consulta(
    con: duckdb.DuckDBPyConnection, sql: str, nome: str, pasta: Path, formato: str, sobrescrever: bool
) -> Path:
    garantir_pasta(pasta)
    if formato == "parquet":
        out = pasta / f"{nome}.parquet"
        if out.exists() and not sobrescrever:
            return out
        con.sql(f"COPY ({sql}) TO '{out.as_posix()}' (FORMAT PARQUET)")
        return out
    if formato == "csv":
        out = pasta / f"{nome}.csv"
        if out.exists() and not sobrescrever:
            return out
        con.sql(f"COPY ({sql}) TO '{out.as_posix()}' (HEADER, DELIMITER ',')")
        return out
    raise ValueError("formato deve ser parquet ou csv")


# Impressão de relação DuckDB
def imprimir_relatorio(titulo: str, rel: duckdb.DuckDBPyRelation, max_linhas: int = 30) -> pd.DataFrame:
    print(f"\n{titulo}")
    df = rel.df()
    if df.empty:
        print("sem linhas")
    else:
        with pd.option_context("display.max_rows", max_linhas, "display.max_columns", None, "display.width", 200):
            print(df)
    return df


# Gravação de pergunta e resposta em arquivo texto
def gravar_qa(base_dir: Path, nome: str, pergunta: str, sql: str, df: pd.DataFrame) -> Path:
    qa_dir = base_dir / "qa"
    garantir_pasta(qa_dir)
    out = qa_dir / f"{nome}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("Pergunta\n")
        f.write(pergunta.strip() + "\n\n")
        f.write("Resposta\n")
        if df.empty:
            f.write("sem linhas\n")
        else:
            f.write(df.to_csv(sep="\t", index=False))
        f.write("\nSQL utilizada\n")
        f.write(sql.strip() + "\n")
    return out


# Execução das consultas exigidas, impressão, exportação e gravação de perguntas e respostas
def executar_consultas(
    con: duckdb.DuckDBPyConnection, pasta: Path, formato: str, topn: int, sobrescrever: bool
) -> list[dict]:
    garantir_pasta(pasta)
    resultados: list[dict] = []

    def rodar(pergunta: str, sql: str, nome: str):
        rel = con.sql(sql)
        df = imprimir_relatorio(pergunta, rel)
        destino = exportar_consulta(con, sql, nome, pasta, formato, sobrescrever)
        qa = gravar_qa(pasta, nome, pergunta, sql, df)
        resultados.append({"nome": nome, "pergunta": pergunta, "linhas": len(df), "export": destino, "qa": qa})

    imprimir_relatorio("Tabelas no DuckDB", con.sql("SHOW TABLES"))
    imprimir_relatorio(
        "Contagem de linhas por tabela",
        con.sql(
            """
            SELECT 'actors' AS tabela, COUNT(*) AS linhas FROM actors
            UNION ALL SELECT 'genres', COUNT(*) FROM genres
            UNION ALL SELECT 'movies', COUNT(*) FROM movies
            UNION ALL SELECT 'tag_names', COUNT(*) FROM tag_names
            UNION ALL SELECT 'tags', COUNT(*) FROM tags
            ORDER BY tabela
            """
        ),
    )

    rodar(
        "Exibir todos os nomes dos filmes em que o Daniel Craig foi escalado, ordenar alfabeticamente pelo título do filme com subconsulta",
        """
        SELECT DISTINCT m.title
        FROM movies m
        WHERE m.mid IN (SELECT a.mid FROM actors a WHERE a.name = 'Daniel Craig')
        ORDER BY m.title
        """,
        "01_daniel_craig_subconsulta",
    )

    rodar(
        "Exibir todos os nomes dos filmes em que o Daniel Craig foi escalado, ordenar alfabeticamente pelo título do filme com inner join",
        """
        SELECT DISTINCT m.title
        FROM movies m
        JOIN actors a ON a.mid = m.mid
        WHERE a.name = 'Daniel Craig'
        ORDER BY m.title
        """,
        "02_daniel_craig_inner_join",
    )

    rodar(
        "Exercício o nome de todo o elenco que fez o filme The Dark Knight",
        """
        SELECT a.name, a.cast_position
        FROM actors a
        JOIN movies m ON m.mid = a.mid
        WHERE m.title = 'The Dark Knight'
        ORDER BY a.cast_position NULLS LAST, a.name
        """,
        "03_elenco_the_dark_knight",
    )

    rodar(
        "Exibir os gêneros distintos e o número de filmes por gênero para gêneros com pelo menos 1000 filmes, ordenado de forma crescente pelo número de filmes",
        """
        SELECT g.genre, COUNT(DISTINCT g.mid) AS num_filmes
        FROM genres g
        GROUP BY g.genre
        HAVING COUNT(DISTINCT g.mid) >= 1000
        ORDER BY num_filmes ASC, g.genre
        """,
        "04_generos_min_1000",
    )

    rodar(
        f"Quais os filmes com maior número de avaliações top {topn}",
        f"""
        SELECT m.title, m.year, m.num_ratings
        FROM movies m
        ORDER BY m.num_ratings DESC NULLS LAST, m.title
        LIMIT {topn}
        """,
        "05_top_filmes_num_avaliacoes",
    )

    rodar(
        "Qual o filme com maior elenco retornando apenas o primeiro considerando desempate por título",
        """
        WITH elenco AS (
          SELECT m.mid, m.title, COUNT(DISTINCT a.name) AS cast_size
          FROM movies m
          JOIN actors a ON a.mid = m.mid
          GROUP BY m.mid, m.title
        )
        SELECT *
        FROM elenco
        ORDER BY cast_size DESC, title
        LIMIT 1
        """,
        "06_filme_maior_elenco",
    )

    rodar(
        "Registro completo do filme The Phantom of the Opera",
        """
        SELECT *
        FROM movies
        WHERE title = 'The Phantom of the Opera'
        """,
        "07_registro_phantom_of_the_opera",
    )

    rodar(
        f"Quais são as tags mais utilizadas top {topn}",
        f"""
        SELECT tn.tag, COUNT(*) AS freq
        FROM tags t
        JOIN tag_names tn ON tn.tid = t.tid
        GROUP BY tn.tag
        ORDER BY freq DESC, tn.tag
        LIMIT {topn}
        """,
        "08_top_tags",
    )

    rodar(
        "Qual ator ou atriz possui o maior número de filmes com papel principal considerando cast_position igual a 1",
        """
        SELECT a.name, COUNT(*) AS filmes_papel_principal
        FROM actors a
        WHERE a.cast_position = 1
        GROUP BY a.name
        ORDER BY filmes_papel_principal DESC, a.name
        LIMIT 1
        """,
        "09_ator_mais_papeis_principais",
    )

    rodar(
        "Qual o gênero com maior nota média sem filtro de quantidade mínima de filmes retornando apenas o primeiro",
        """
        SELECT g.genre, AVG(m.rating) AS avg_rating, COUNT(*) AS n_registros
        FROM genres g
        JOIN movies m ON m.mid = g.mid
        GROUP BY g.genre
        ORDER BY avg_rating DESC, n_registros DESC, g.genre
        LIMIT 1
        """,
        "10_genero_maior_media",
    )

    return resultados


# Validação final dos requisitos e registro em arquivo
def validar_requisitos(
    resultados: list[dict],
    tabelas_exportadas: list[Path],
    db_path: Path,
    export_dir: Path,
) -> None:
    linhas = []
    ok_db = db_path.exists()
    linhas.append(f"Banco DuckDB existente: {'OK' if ok_db else 'FALHA'}  {db_path}")

    for p in tabelas_exportadas:
        linhas.append(f"Tabela exportada: {'OK' if p.exists() else 'FALHA'}  {p}")

    for r in resultados:
        linhas.append(f"Consulta exportada: {'OK' if Path(r['export']).exists() else 'FALHA'}  {r['export']}")
        linhas.append(f"Arquivo de pergunta e resposta: {'OK' if Path(r['qa']).exists() else 'FALHA'}  {r['qa']}")
        if r["linhas"] >= 0:
            linhas.append(f"Linhas retornadas na consulta {r['nome']}: {r['linhas']}")

    relatorio = "\n".join(linhas)
    print("\nVerificação de requisitos")
    print(relatorio)

    garantir_pasta(export_dir)
    with open(export_dir / "relatorio_requisitos.txt", "w", encoding="utf-8") as f:
        f.write(relatorio + "\n")


# Função principal
def main() -> None:
    data_dir = DATA_DIR
    garantir_entradas(data_dir)

    movies, actors, genres, tags, tag_names = carregar_dados(data_dir)

    con = conectar_persistente(DB_FILE)
    tabelas_exportadas: list[Path] = []
    try:
        materializar(con, movies, actors, genres, tags, tag_names)

        for tabela in ("movies", "actors", "genres", "tags", "tag_names"):
            destino = exportar_tabela(con, tabela, EXPORT_DIR / "tabelas", EXPORT_FORMAT, OVERWRITE)
            tabelas_exportadas.append(destino)
            print(f"Tabela exportada  {tabela}  {destino}")

        resultados = executar_consultas(con, EXPORT_DIR / "consultas", EXPORT_FORMAT, TOPN, OVERWRITE)

        con.sql("CHECKPOINT")
        print(f"\nConcluído banco de dados DuckDB  {DB_FILE}")
        print(f"Arquivos de exportação em  {EXPORT_DIR}")

        validar_requisitos(resultados, tabelas_exportadas, DB_FILE, EXPORT_DIR)
    finally:
        con.close()


if __name__ == "__main__":
    main()
