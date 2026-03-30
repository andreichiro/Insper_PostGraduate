from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
from operator import add

spark = SparkSession.builder.master("local[*]").appName("DNA_Compare_2025").getOrCreate()
sc = spark.sparkContext

# --- Domain constants ---
IUPAC = set(list("ACGTNRYKMSWBDHV"))  # canonical + ambiguous
CANON = set(list("ACGT"))
PURINES = set("AG")
PYRIMIDINES = set("CT")

# --- Utilities ---
def fasta_bases(sc, path, allowed=IUPAC):
    """
    Read a FASTA file and return an RDD[str] of per-base characters (A/C/G/T/ambiguous),
    with headers removed and everything normalized to uppercase.
    """
    return (
        sc.textFile(path)
          .filter(lambda line: not line.startswith(">"))   # drop headers
          .map(lambda line: line.strip().upper())
          .filter(lambda line: line != "")
          .flatMap(list)                                   # char iterator
          .filter(lambda c: c in allowed)                  # keep only IUPAC bases
    )

def base_counts_and_freqs(base_rdd):
    """
    Return (counts_rdd, total_bases:int, freqs_rdd) where:
      - counts_rdd: RDD[(base, count)]
      - freqs_rdd:  RDD[(base, pct_float)]
    Ensures a stable schema by emitting zeroes for absent symbols.
    """
    counts = base_rdd.map(lambda c: (c, 1)).reduceByKey(add)
    total = counts.values().sum()
    # Fill missing bases with zeros to keep a stable set of keys
    counts_full = sc.parallelize(list(IUPAC)).map(lambda b: (b, 0)).union(counts).reduceByKey(add)
    freqs = counts_full.map(lambda kv: (kv[0], (kv[1] / total * 100.0) if total else 0.0))
    return counts_full, int(total), freqs

def gc_content_from_counts(counts_rdd):
    cd = dict(counts_rdd.collect())
    a, c, g, t = cd.get('A', 0), cd.get('C', 0), cd.get('G', 0), cd.get('T', 0)
    denom = a + c + g + t
    return (g + c) / denom if denom else 0.0

def indexed(base_rdd):
    """Annotate each base with its zero-based index: returns RDD[(idx:int, base:str)]."""
    return base_rdd.zipWithIndex().map(lambda cb: (cb[1], cb[0]))

def compare_two_fastas(sc, path_a, path_b):
    """
    Full analysis + comparison between two FASTA files.
    Returns a dict of metrics and samples suitable for printing or further use.
    """
    # Read & cache bases
    bases_a = fasta_bases(sc, path_a).persist(StorageLevel.MEMORY_ONLY)
    bases_b = fasta_bases(sc, path_b).persist(StorageLevel.MEMORY_ONLY)

    # Per-sample stats
    counts_a, total_a, freqs_a = base_counts_and_freqs(bases_a)
    counts_b, total_b, freqs_b = base_counts_and_freqs(bases_b)
    gc_a = gc_content_from_counts(counts_a)
    gc_b = gc_content_from_counts(counts_b)

    # Index for positional comparison
    idx_a = indexed(bases_a)
    idx_b = indexed(bases_b)

    # Join on index; fullOuterJoin protects against unequal lengths
    joined = idx_a.fullOuterJoin(idx_b)  # (idx, (a_base_or_None, b_base_or_None))

    # Basic tallies
    def classify(rec):
        idx, (a, b) = rec
        if a is None or b is None:
            return ("Gap_or_length_mismatch", 1)
        if a == b and a in CANON:
            return ("Match_canonical", 1)
        if a == b and a not in CANON:
            return ("Match_ambiguous", 1)
        if a == 'N' or b == 'N':
            return ("N_involved", 1)
        if a not in CANON or b not in CANON:
            return ("Ambiguous_mismatch", 1)
        # canonical mismatch: transition or transversion?
        if (a in PURINES and b in PURINES) or (a in PYRIMIDINES and b in PYRIMIDINES):
            return ("Transition", 1)
        return ("Transversion", 1)

    summary = joined.map(classify).reduceByKey(add)

    # Comparable subset: both canonical (A/C/G/T)
    comparable = (
        joined
        .map(lambda rec: rec[1])
        .filter(lambda ab: ab[0] in CANON and ab[1] in CANON)
        .cache()
    )
    matches = comparable.filter(lambda ab: ab[0] == ab[1]).count()
    comparable_positions = comparable.count()
    pid = (matches / comparable_positions) if comparable_positions else 0.0  # percent identity (0..1)

    # Per-change spectrum and a small sample of variant sites
    changes = (
        joined
        .filter(lambda rec: rec[1][0] in CANON and rec[1][1] in CANON and rec[1][0] != rec[1][1])
        .map(lambda rec: (f"{rec[1][0]}->{rec[1][1]}", 1))
        .reduceByKey(add)
    )
    # First 20 mismatches as (pos, A, B)
    mismatch_sample = (
        joined
        .filter(lambda rec: rec[1][0] in IUPAC and rec[1][1] in IUPAC and rec[1][0] != rec[1][1])
        .map(lambda rec: (int(rec[0]), rec[1][0], rec[1][1]))
        .sortBy(lambda t: t[0])
        .take(20)
    )

    return {
        "totals": {"A": total_a, "B": total_b},
        "gc": {"A": gc_a, "B": gc_b},
        "counts_A": sorted(counts_a.collect(), key=lambda kv: kv[0]),
        "counts_B": sorted(counts_b.collect(), key=lambda kv: kv[0]),
        "freqs_A_pct": sorted(freqs_a.collect(), key=lambda kv: kv[0]),
        "freqs_B_pct": sorted(freqs_b.collect(), key=lambda kv: kv[0]),
        "summary": dict(summary.collect()),
        "percent_identity_canonical": pid,
        "comparable_positions": comparable_positions,
        "change_spectrum": dict(changes.collect()),
        "mismatch_sample_first20": mismatch_sample,
    }

# ---------------------------
# Example usage (plug in your files here)
# ---------------------------
base_path = "../../dados/10_dados/covid_dna/"
file_first  = base_path + "SARS-CoV-2-Wuhan-NC_045512.2.fasta"          # "first"
file_second = base_path + "SARS-CoV-2-Washington_MT293201.1.fasta"       # "second" or your new file

report = compare_two_fastas(sc, file_first, file_second)

# Pretty-print a concise summary
def pct(x): return f"{x*100:.4f}%"
print("Total bases (A/B):", report["totals"]["A"], "/", report["totals"]["B"])
print("GC% (A/B):", f"{report['gc']['A']*100:.3f}%", "/", f"{report['gc']['B']*100:.3f}%")
print("Percent identity (canonical only):", pct(report["percent_identity_canonical"]))
print("Comparable positions (A|B canonical):", report["comparable_positions"])
print("Transitions vs Transversions:", 
      "Ti:", report["summary"].get("Transition", 0), 
      "Tv:", report["summary"].get("Transversion", 0))
print("Change spectrum (top 10):", sorted(report["change_spectrum"].items(), key=lambda x: -x[1])[:10])
print("First 20 mismatches (pos, A, B):")
for rec in report["mismatch_sample_first20"]:
    print(rec)
