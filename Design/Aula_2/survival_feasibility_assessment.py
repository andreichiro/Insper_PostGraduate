#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd


@dataclass
class Thresholds:
    max_event_rate_for_effective: float = 0.95
    min_censored_for_effective: int = 5000
    max_share_unmatched_entries: float = 0.50
    max_share_unmatched_interactions: float = 0.50
    max_estado_missing: float = 0.35
    max_utm_missing: float = 0.15


class SurvivalFeasibilityAssessor:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.data_dir = base_dir / "base_aprendizap"
        self.out_dir = base_dir / "analysis_output" / "survival_assessment"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = Thresholds()

    @staticmethod
    def _escape(path: Path) -> str:
        return str(path).replace("'", "''")

    def _connect(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect()
        conn.execute("PRAGMA threads=4")
        conn.execute(
            f"CREATE VIEW dim_teachers AS SELECT * FROM read_csv('{self._escape(self.data_dir / 'dim_teachers.csv')}', delim=';', header=true, ignore_errors=true)"
        )
        conn.execute(
            f"CREATE VIEW entries AS SELECT * FROM read_csv_auto('{self._escape(self.data_dir / 'fct_teachers_entries.csv')}', header=true)"
        )
        conn.execute(
            f"CREATE VIEW interactions AS SELECT * FROM read_csv_auto('{self._escape(self.data_dir / 'fct_teachers_contents_interactions.csv')}', header=true)"
        )
        return conn

    def _query_tables(self, conn: duckdb.DuckDBPyConnection) -> Dict[str, pd.DataFrame]:
        tables: Dict[str, pd.DataFrame] = {}

        tables["counts"] = conn.execute(
            """
            SELECT 'dim_teachers' AS table_name, COUNT(*) AS rows, COUNT(DISTINCT unique_id) AS unique_users FROM dim_teachers
            UNION ALL
            SELECT 'entries', COUNT(*), COUNT(DISTINCT unique_id) FROM entries
            UNION ALL
            SELECT 'interactions', COUNT(*), COUNT(DISTINCT unique_id) FROM interactions
            """
        ).fetchdf()

        tables["coverage"] = conn.execute(
            """
            SELECT 'entries' AS table_name, MIN(data_inicio) AS min_ts, MAX(data_fim) AS max_ts FROM entries
            UNION ALL
            SELECT 'interactions', MIN(data_inicio), MAX(data_inicio) FROM interactions
            UNION ALL
            SELECT 'dim_teachers_data_entrada', MIN(data_entrada), MAX(data_entrada) FROM dim_teachers
            """
        ).fetchdf()

        tables["identity_match"] = conn.execute(
            """
            WITH e AS (
                SELECT COUNT(*) AS entries_rows,
                       SUM(CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END) AS entries_rows_matched
                FROM entries e LEFT JOIN dim_teachers d ON e.unique_id=d.unique_id
            ),
            i AS (
                SELECT COUNT(*) AS interactions_rows,
                       SUM(CASE WHEN d.unique_id IS NOT NULL THEN 1 ELSE 0 END) AS interactions_rows_matched
                FROM interactions i LEFT JOIN dim_teachers d ON i.unique_id=d.unique_id
            )
            SELECT
                entries_rows,
                entries_rows_matched,
                interactions_rows,
                interactions_rows_matched,
                1.0 - entries_rows_matched::DOUBLE / NULLIF(entries_rows,0) AS entries_unmatched_share,
                1.0 - interactions_rows_matched::DOUBLE / NULLIF(interactions_rows,0) AS interactions_unmatched_share
            FROM e CROSS JOIN i
            """
        ).fetchdf()

        tables["event_windows"] = conn.execute(
            """
            WITH snap AS (
                SELECT GREATEST((SELECT MAX(data_fim) FROM entries),(SELECT MAX(data_inicio) FROM interactions)) AS snapshot_ts
            ),
            act AS (
                SELECT d.unique_id,d.data_entrada,
                       GREATEST(COALESCE(MAX(e.data_fim), d.data_entrada), COALESCE(MAX(i.data_inicio), d.data_entrada), d.data_entrada) AS last_activity,
                       s.snapshot_ts
                FROM dim_teachers d
                CROSS JOIN snap s
                LEFT JOIN entries e ON d.unique_id=e.unique_id
                LEFT JOIN interactions i ON d.unique_id=i.unique_id
                GROUP BY d.unique_id,d.data_entrada,s.snapshot_ts
            ),
            base AS (
                SELECT *,
                  (epoch(snapshot_ts)-epoch(last_activity))/86400.0 AS inactivity_days,
                  (epoch(snapshot_ts)-epoch(data_entrada))/86400.0 AS exposure_days
                FROM act
            ),
            w AS (SELECT 7 AS wnd UNION ALL SELECT 14 UNION ALL SELECT 30 UNION ALL SELECT 60 UNION ALL SELECT 90)
            SELECT
              wnd AS inactivity_window_days,
              COUNT(*) AS total_teachers,
              SUM(CASE WHEN exposure_days>=wnd THEN 1 ELSE 0 END) AS at_risk,
              SUM(CASE WHEN exposure_days>=wnd AND inactivity_days>wnd THEN 1 ELSE 0 END) AS events,
              SUM(CASE WHEN exposure_days>=wnd AND inactivity_days<=wnd THEN 1 ELSE 0 END) AS censored,
              SUM(CASE WHEN exposure_days>=wnd AND inactivity_days>wnd THEN 1 ELSE 0 END)::DOUBLE / NULLIF(SUM(CASE WHEN exposure_days>=wnd THEN 1 ELSE 0 END),0) AS event_rate
            FROM base CROSS JOIN w
            GROUP BY wnd
            ORDER BY wnd
            """
        ).fetchdf()

        tables["event_time_distribution_30d"] = conn.execute(
            """
            WITH snap AS (
              SELECT GREATEST((SELECT MAX(data_fim) FROM entries),(SELECT MAX(data_inicio) FROM interactions)) AS snapshot_ts
            ),
            act AS (
              SELECT d.unique_id,d.data_entrada,
                     GREATEST(COALESCE(MAX(e.data_fim), d.data_entrada), COALESCE(MAX(i.data_inicio), d.data_entrada), d.data_entrada) AS last_activity,
                     s.snapshot_ts
              FROM dim_teachers d
              CROSS JOIN snap s
              LEFT JOIN entries e ON d.unique_id=e.unique_id
              LEFT JOIN interactions i ON d.unique_id=i.unique_id
              GROUP BY d.unique_id,d.data_entrada,s.snapshot_ts
            ),
            calc AS (
              SELECT
                (epoch(snapshot_ts)-epoch(last_activity))/86400.0 AS inactivity_days,
                (epoch(last_activity)+30*86400-epoch(data_entrada))/86400.0 AS event_time_days
              FROM act
            )
            SELECT
              COUNT(*) AS n_events,
              MEDIAN(event_time_days) AS median_event_time_days,
              QUANTILE(event_time_days,0.1) AS p10,
              QUANTILE(event_time_days,0.25) AS p25,
              QUANTILE(event_time_days,0.75) AS p75,
              QUANTILE(event_time_days,0.9) AS p90
            FROM calc
            WHERE inactivity_days>30
            """
        ).fetchdf()

        tables["reactivation_gaps"] = conn.execute(
            """
            WITH activity AS (
              SELECT e.unique_id, e.data_fim AS ts FROM entries e JOIN dim_teachers d ON e.unique_id=d.unique_id
              UNION ALL
              SELECT i.unique_id, i.data_inicio AS ts FROM interactions i JOIN dim_teachers d ON i.unique_id=d.unique_id
            ),
            ord AS (
              SELECT unique_id, ts,
                     LAG(ts) OVER(PARTITION BY unique_id ORDER BY ts) AS prev_ts
              FROM activity
            ),
            g AS (
              SELECT unique_id,
                     CASE WHEN prev_ts IS NULL THEN NULL ELSE (epoch(ts)-epoch(prev_ts))/86400.0 END AS gap_days
              FROM ord
            ),
            agg AS (
              SELECT unique_id,
                     COUNT(*) AS n_events,
                     SUM(CASE WHEN gap_days>30 THEN 1 ELSE 0 END) AS gaps_gt30,
                     SUM(CASE WHEN gap_days>60 THEN 1 ELSE 0 END) AS gaps_gt60,
                     MAX(COALESCE(gap_days,0)) AS max_gap_days
              FROM g
              GROUP BY unique_id
            )
            SELECT
              COUNT(*) AS teachers_with_any_activity,
              SUM(CASE WHEN n_events>=2 THEN 1 ELSE 0 END) AS teachers_with_2plus_events,
              SUM(CASE WHEN gaps_gt30>0 THEN 1 ELSE 0 END) AS teachers_with_reactivation_after_30d_gap,
              SUM(CASE WHEN gaps_gt30>=2 THEN 1 ELSE 0 END) AS teachers_with_multiple_30d_gaps,
              SUM(CASE WHEN gaps_gt60>0 THEN 1 ELSE 0 END) AS teachers_with_reactivation_after_60d_gap,
              MEDIAN(max_gap_days) AS median_max_gap_days,
              QUANTILE(max_gap_days,0.9) AS p90_max_gap_days
            FROM agg
            """
        ).fetchdf()

        tables["covariate_missingness"] = conn.execute(
            """
            SELECT
              SUM(CASE WHEN estado IS NULL OR TRIM(estado)='' THEN 1 ELSE 0 END)::DOUBLE/COUNT(*) AS estado_missing,
              SUM(CASE WHEN utm_origin IS NULL OR TRIM(utm_origin)='' THEN 1 ELSE 0 END)::DOUBLE/COUNT(*) AS utm_missing,
              SUM(CASE WHEN currentstage IS NULL OR TRIM(currentstage)='' THEN 1 ELSE 0 END)::DOUBLE/COUNT(*) AS stage_missing
            FROM dim_teachers
            """
        ).fetchdf()

        return tables

    def _decision(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
        win30 = tables["event_windows"].loc[tables["event_windows"]["inactivity_window_days"] == 30].iloc[0]
        identity = tables["identity_match"].iloc[0]
        missing = tables["covariate_missingness"].iloc[0]

        possible = True
        reasons: List[str] = []
        risks: List[str] = []

        if int(win30["at_risk"]) == 0:
            possible = False
            reasons.append("Sem população em risco para janela de 30 dias.")
        if int(win30["events"]) == 0:
            possible = False
            reasons.append("Sem eventos de não-atividade para modelagem de tempo até evento.")

        effective = possible
        if float(win30["event_rate"]) > self.thresholds.max_event_rate_for_effective:
            effective = False
            risks.append(
                f"Evento muito prevalente na janela 30d ({float(win30['event_rate']):.4f}), baixa separação entre evento e censura."
            )
        if int(win30["censored"]) < self.thresholds.min_censored_for_effective:
            effective = False
            risks.append(
                f"Poucos censurados ({int(win30['censored'])}), reduz robustez para curvas comparativas e modelagem preditiva."
            )
        if float(identity["entries_unmatched_share"]) > self.thresholds.max_share_unmatched_entries:
            effective = False
            risks.append(
                f"Alta fração de sessões sem identidade docente ({float(identity['entries_unmatched_share']):.2%})."
            )
        if float(identity["interactions_unmatched_share"]) > self.thresholds.max_share_unmatched_interactions:
            effective = False
            risks.append(
                f"Alta fração de interações sem identidade docente ({float(identity['interactions_unmatched_share']):.2%})."
            )
        if float(missing["estado_missing"]) > self.thresholds.max_estado_missing:
            risks.append(f"Missing alto em estado ({float(missing['estado_missing']):.2%}).")
        if float(missing["utm_missing"]) > self.thresholds.max_utm_missing:
            risks.append(f"Missing alto em utm_origin ({float(missing['utm_missing']):.2%}).")

        recommendation = (
            "Sobrevivência é possível, mas não é efetiva no formato simples de 'tempo até ficar inativo uma vez'. "
            "Preferir modelo de eventos recorrentes (retorno/abandono) em painel mensal ou hazard discreto com covariáveis dinâmicas."
            if possible and not effective
            else "Sobrevivência é possível e efetiva para uso operacional."
            if possible and effective
            else "Com os dados atuais, sobrevivência não é viável."
        )

        return {
            "possible": possible,
            "effective_now": effective,
            "blocking_reasons": reasons,
            "major_risks": risks,
            "recommendation": recommendation,
        }

    def _write_outputs(self, tables: Dict[str, pd.DataFrame], decision: Dict[str, object]) -> None:
        for name, df in tables.items():
            df.to_csv(self.out_dir / f"{name}.csv", index=False)

        summary = {
            "decision": decision,
            "paths": {
                "output_dir": str(self.out_dir),
                "script": str(self.base_dir / "survival_feasibility_assessment.py"),
            },
        }
        (self.out_dir / "survival_feasibility_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        win30 = tables["event_windows"].loc[tables["event_windows"]["inactivity_window_days"] == 30].iloc[0]
        identity = tables["identity_match"].iloc[0]
        evt = tables["event_time_distribution_30d"].iloc[0]
        react = tables["reactivation_gaps"].iloc[0]
        miss = tables["covariate_missingness"].iloc[0]

        md = f"""# Survival Feasibility Assessment

## Decisão
- possible: `{decision['possible']}`
- effective_now: `{decision['effective_now']}`
- recommendation: {decision['recommendation']}

## Evidências chave
- Janela 30d: `events={int(win30['events'])}`, `censored={int(win30['censored'])}`, `event_rate={float(win30['event_rate']):.4f}`
- Unmatched identity: `entries={float(identity['entries_unmatched_share']):.2%}`, `interactions={float(identity['interactions_unmatched_share']):.2%}`
- Concentração do tempo de evento (30d): `p25={float(evt['p25']):.2f}d`, `mediana={float(evt['median_event_time_days']):.2f}d`
- Reativação após gap>30d: `{int(react['teachers_with_reactivation_after_30d_gap'])}` professores
- Missing covariáveis: `estado={float(miss['estado_missing']):.2%}`, `utm_origin={float(miss['utm_missing']):.2%}`

## Interpretação
- Há estrutura temporal suficiente para modelagem de sobrevivência.
- No entanto, a definição simples de evento de não-atividade produz evento quase universal e poucos censurados.
- O dado indica comportamento recorrente (abandono/retorno), então um modelo single-event perde informação relevante.
"""
        (self.out_dir / "survival_feasibility_details.md").write_text(md, encoding="utf-8")

    def run(self) -> Dict[str, object]:
        conn = self._connect()
        try:
            tables = self._query_tables(conn)
            decision = self._decision(tables)
            self._write_outputs(tables, decision)
            return decision
        finally:
            conn.close()


def main() -> None:
    base_dir = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
    assessor = SurvivalFeasibilityAssessor(base_dir=base_dir)
    decision = assessor.run()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

