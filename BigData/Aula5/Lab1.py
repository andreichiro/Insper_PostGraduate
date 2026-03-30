"""
André Katsurada - Análise de logs ssh c/ rdds do spark

Perguntas
1) Quantas linhas há no arquivo de log
2) Quantos logins com sucesso ocorreram no sistema
3) Quais são os usuários que logaram neste sistema
4) Top 10 usuários que acessam a máquina com maior frequência, exceto root
5) Quais ips estão acessando mais frequentemente esta máquina

Decisões
- considera login com sucesso a linha com padrão:
  "Accepted <método> for <user> from <ip>"
- os usuários que logaram são extraídos dessas mesmas linhas accepted
- a contagem de ip considera eventos sshd de acesso ou tentativa:
  "Accepted", "Invalid user", "Failed password", "Did not receive identification string",
  "Bad protocol version identification", "maximum authentication attempts exceeded",
  "Received disconnect from", "Connection closed by", "Connection reset by", "fatal: Unable to negotiate"
- ignora "Disconnected from" para evitar duplicidade
- ignora ip 0.0.0.0 e processa apenas ipv4
- tudo é feito por rdds, sem spark sql
"""

from __future__ import annotations

import html
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

from pyspark.sql import SparkSession


# path
LOG_PATH = "/Users/akatsurada/Documents/INSPER/BigData/Aula5/auth.log"

# Params
APP_NAME = "SSHLogRDD"
MASTER = "local[*]"
TOP_K_IPS = 20
HTML_OUT = "./ssh_rdd_report.html"


# Utils

# Regex p/ IPv4
IPV4_RE = re.compile(r"\b((?:\d{1,3}\.){3}\d{1,3})\b")

# Métodos de auth + comuns
ACCEPTED_METHOD_PREFIXES = (
    "Accepted password ",
    "Accepted publickey ",
    "Accepted keyboard-interactive ",
    "Accepted keyboard-interactive/pam ",
    "Accepted gssapi-with-mic ",
)

# Prefixos de eventos p/ tentativas ou acessos
SSH_EVENT_PREFIXES_FOR_IP = (
    "Accepted ",
    "Invalid user ",
    "Failed password for ",
    "Did not receive identification string from ",
    "Bad protocol version identification ",
    "error: maximum authentication attempts exceeded",
    "Received disconnect from ",
    "Connection closed by ",
    "Connection reset by ",
    "fatal: Unable to negotiate",
)

DISCONNECTED_PREFIX = "Disconnected from "

@dataclass(frozen=True)
class AnalysisResult:
    """Estrutura com todas as respostas consolidadas"""
    total_lines: int
    successful_logins: int
    logged_users: List[str]
    top_users_non_root: List[Tuple[str, int]]
    ip_freq: List[Tuple[str, int]]

class RDDAnalysis:
    """
    Usar RDDs p/ 
    - ler o log
    - usar regex simples
    - usar mapPartitions para minimizar overhead
    - usar reduceByKey p/ contagens e takeOrdered para top-k eficiente
    """

    @staticmethod
    def _extract_message_if_sshd(line: str) -> Optional[str]:
        """
        Extrai a parte message se e somente se a linha for do sshd
        - Encontra o primeiro ': ' que separa cabeçalho de message
        - Retrocede desde esse ponto para localizar o token do programa (ex: 'sshd[1361]')
        - Verifica se o programa é 'sshd', ignorando o pid
        """
        cut = line.find(": ")
        if cut < 0:
            return None
        i = cut - 1
        while i >= 0 and line[i] != " ":
            i -= 1
        program_token = line[i + 1:cut]  # exemplo: 'sshd[1361]' ou 'sshd'
        program_name = program_token.split("[", 1)[0]
        if program_name != "sshd":
            return None
        return line[cut + 2:]

    @staticmethod
    def _is_accepted(msg: str) -> bool:
        """Verifica se a mensagem é de login accepted por prefixo fixo"""
        return any(msg.startswith(p) for p in ACCEPTED_METHOD_PREFIXES)

    @staticmethod
    def _extract_user_ip_from_accepted(msg: str) -> Optional[Tuple[str, str]]:
        """
        Extrai (user, ip) de uma mensagem accepted
        pegando por ' for ' e ' from '
        e valida o ip com regex curta
        """
        i_for = msg.find(" for ")
        if i_for < 0:
            return None
        i_from = msg.find(" from ", i_for + 5)
        if i_from < 0:
            return None
        user = msg[i_for + 5:i_from].strip()
        if not user:
            return None
        rest = msg[i_from + 6:]
        j_port = rest.find(" port ")
        candidate = rest if j_port < 0 else rest[:j_port]
        m = IPV4_RE.search(candidate)
        if not m:
            return None
        ip = m.group(1)
        if ip == "0.0.0.0":
            return None
        return user, ip

    @staticmethod
    def _extract_ip_from_event(msg: str) -> Optional[str]:
        """
        Extrai ip de uma mensagem de evento sshd representativa
        retorna none para 0.0.0.0 ou quando ñ encontra ipv4
        """
        m = IPV4_RE.search(msg)
        if not m:
            return None
        ip = m.group(1)
        if ip == "0.0.0.0":
            return None
        return ip

    def analyze(self, spark: SparkSession, log_path: str, top_k_ips: int) -> AnalysisResult:
        """
        Executa:
        1) contagem de linhas totais
        2) filtra e mapeia logins accepted -> rdd com (user, ip)
        3) deriva usuários e contagem por usuário
        4) filtra root e calcula top 10 por frequência (empates por nome)
        5) extrai ips relevantes e calcula top-k
        """
        sc = spark.sparkContext
        raw = sc.textFile(log_path).cache()

        try:
            # 1) total de linhas
            total_lines = raw.count()

            # 2) rdd de (user, ip) apenas a partir de mensagens accepted do sshd
            def per_part_accepteds(lines: Iterable[str]):
                for line in lines:
                    msg = self._extract_message_if_sshd(line)
                    if not msg:
                        continue
                    if self._is_accepted(msg):
                        pair = self._extract_user_ip_from_accepted(msg)
                        if pair:
                            yield pair

            accepted_rdd = raw.mapPartitions(per_part_accepteds).cache()
            successful_logins = accepted_rdd.count()

            # 3) usuários distintos ordenados
            logged_users = accepted_rdd.map(lambda ui: ui[0]).distinct().collect()
            logged_users.sort()

            # 4) top 10 usuários, exceto root
            # usa takeOrdered com chave (-contagem, usuário) para evitar sort global custoso
            from operator import add
            user_counts = accepted_rdd.map(lambda ui: (ui[0], 1)).reduceByKey(add)
            top_users_non_root = user_counts \
                .filter(lambda kv: kv[0].lower() != "root") \
                .takeOrdered(10, key=lambda kv: (-kv[1], kv[0]))

            # 5) ranking de ip a partir de eventos representativos do sshd
            def per_part_ips(lines: Iterable[str]):
                for line in lines:
                    msg = self._extract_message_if_sshd(line)
                    if not msg:
                        continue
                    if msg.startswith(DISCONNECTED_PREFIX):
                        # ignorado para ñ duplicar c/ received disconnect
                        continue
                    if not any(msg.startswith(p) for p in SSH_EVENT_PREFIXES_FOR_IP):
                        continue
                    ip = self._extract_ip_from_event(msg)
                    if ip:
                        yield (ip, 1)

            ip_freq = raw.mapPartitions(per_part_ips) \
                         .reduceByKey(lambda a, b: a + b) \
                         .takeOrdered(int(top_k_ips), key=lambda kv: (-kv[1], kv[0]))

            return AnalysisResult(
                total_lines=int(total_lines),
                successful_logins=int(successful_logins),
                logged_users=logged_users,
                top_users_non_root=[(u, int(n)) for (u, n) in top_users_non_root],
                ip_freq=[(ip, int(cnt)) for (ip, cnt) in ip_freq],
            )
        finally:
            try:
                raw.unpersist(False)
            except Exception:
                pass

class HTMLReport:
    """
    Construtor p/ html simples c/ as respostas
    """

    def __init__(self, out_path: str, title: str) -> None:
        self.out_path = out_path
        self.title = title
        self.parts: List[str] = []

    def begin(self, log_path: str, spark_version: str, assumptions: List[str]) -> None:
        self.parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        self.parts.append(f"<title>{html.escape(self.title)}</title>")
        self.parts.append(
            "<style>"
            "body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#222}"
            "h1,h2,h3{margin:1em 0 .5em}p{margin:.6em 0;line-height:1.4}"
            "table{border-collapse:collapse;width:100%;margin:8px 0 16px}"
            "th,td{border:1px solid #ddd;padding:6px 8px}th{text-align:left;background:#f7f7f7}"
            "td.num{text-align:right;font-variant-numeric:tabular-nums}"
            ".note{background:#f5f9ff;border-left:4px solid #8bb9ff;padding:.6em .8em;margin:.6em 0}"
            "</style></head><body>"
        )
        self.parts.append(f"<h1>{html.escape(self.title)}</h1>")
        self.parts.append("<div class='note'>")
        self.parts.append(f"<p>arquivo: {html.escape(log_path)}</p>")
        self.parts.append(f"<p>spark: {html.escape(spark_version)}</p>")
        self.parts.append(f"<p>gerado em: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>")
        self.parts.append("</div>")
        if assumptions:
            self.parts.append("<h2>premissas</h2><ul>")
            for a in assumptions:
                self.parts.append(f"<li>{html.escape(a)}</li>")
            self.parts.append("</ul>")

    def section_answers(self, result: AnalysisResult, top_k_ips: int) -> None:
        self.parts.append("<h2>resultados com rdds</h2>")
        q = [
            ("quantas linhas há no arquivo de log", str(result.total_lines)),
            ("quantos logins com sucesso ocorreram no sistema", str(result.successful_logins)),
            ("quais são os usuários que logaram neste sistema",
             ", ".join(result.logged_users) if result.logged_users else "(nenhum)"),
        ]
        for question, answer in q:
            self.parts.append(f"<h3>{html.escape(question)}</h3><p>{html.escape(answer)}</p>")

        self.parts.append("<h3>usuários mais frequentes, exceto root (top 10)</h3>")
        self.parts.append("<table><thead><tr><th>usuário</th><th>logins</th></tr></thead><tbody>")
        if result.top_users_non_root:
            for u, n in result.top_users_non_root:
                self.parts.append(f"<tr><td>{html.escape(u)}</td><td class='num'>{int(n)}</td></tr>")
        else:
            self.parts.append("<tr><td colspan='2'>(nenhum)</td></tr>")
        self.parts.append("</tbody></table>")

        self.parts.append(f"<h3>ips que mais acessam esta máquina, top {int(top_k_ips)}</h3>")
        self.parts.append("<table><thead><tr><th>ip</th><th>eventos</th></tr></thead><tbody>")
        for ip, cnt in result.ip_freq:
            self.parts.append(f"<tr><td>{html.escape(ip)}</td><td class='num'>{int(cnt)}</td></tr>")
        self.parts.append("</tbody></table>")

    def end(self) -> None:
        self.parts.append("</body></html>")
        content = "".join(self.parts)
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write(content)


def get_spark(app_name: str, master: str) -> SparkSession:
    """
    Spark session local
    """
    return SparkSession.builder.appName(app_name).master(master).getOrCreate()


def print_answers(res: AnalysisResult, top_k: int) -> None:
    """
    Print p/ inspeção rápida
    """
    print("resultados com rdds")
    print(f"1) total de linhas: {res.total_lines}")
    print(f"2) logins com sucesso: {res.successful_logins}")
    print(f"3) usuários que logaram: {', '.join(res.logged_users) if res.logged_users else '(nenhum)'}")
    print("4) usuários mais frequentes (exceto root), top 10:")
    if res.top_users_non_root:
        for u, n in res.top_users_non_root:
            print(f"   {u}: {n}")
    else:
        print("   (nenhum)")
    print(f"5) ips mais frequentes, top {int(top_k)}:")
    for ip, cnt in res.ip_freq:
        print(f"   {ip}: {cnt}")


def main() -> None:
    """
    Ponto de entrada 
    """
    # Valida path 
    if not os.path.exists(LOG_PATH) or not os.path.isfile(LOG_PATH):
        print(f"erro: arquivo não encontrado: {LOG_PATH}", file=sys.stderr)
        sys.exit(1)

    spark = get_spark(APP_NAME, MASTER)
    try:
        print(f"analisando: {LOG_PATH}")

        analysis = RDDAnalysis()
        res = analysis.analyze(spark, LOG_PATH, TOP_K_IPS)

        print_answers(res, TOP_K_IPS)

        assumptions = [
            "login com sucesso identificado por 'Accepted <método> for <user> from <ip>'",
            "usuários que logaram são extraídos dessas mesmas linhas 'Accepted'",
            "ranking de ip ignora 'Disconnected from' para evitar duplicidade",
            "apenas ipv4 e ip 0.0.0.0 é ignorado",
        ]

        report = HTMLReport(HTML_OUT, "relatório de acesso ssh (rdds)")
        report.begin(LOG_PATH, spark.version, assumptions)
        report.section_answers(res, TOP_K_IPS)
        report.end()
        print(f"html gerado em: {HTML_OUT}")
    except Exception as exc:
        print(f"erro: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
