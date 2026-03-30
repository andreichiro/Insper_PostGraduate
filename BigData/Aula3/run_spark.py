# runner_vscode.py
from top_cited_patents_strategies import run

# Choose engine: "rdd" | "df" | "sql"
run({
    "ENGINE": "df",
    "INPUT_PATH": "data/citations.csv",  # custom path here

    # Optional:
    # "OUTPUT_PATH": "out/top10_df",     # write instead of stdout
    # "OUTPUT_FORMAT": "csv",            # "csv" | "parquet"
    # "TOP_N": 10,
    # "VERBOSITY": 1,                    # 0 WARN | 1 INFO | 2 DEBUG
})
