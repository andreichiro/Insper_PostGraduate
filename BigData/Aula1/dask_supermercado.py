import dask
import dask.dataframe as dd
import pandas as pd
from dask import visualize

df = dd.read_csv(
    '/Users/akatsurada/Documents/INSPER/BigData/Aula1/supermercado (2).csv',
    sep=';'
)

print("Primeiras linhas do dataset:")
print(df.head(n=5, npartitions=-1))

# 1) Parse em pandas (para este exemplo em memória) e padronização de nomes
pdf = pd.read_csv('/Users/akatsurada/Documents/INSPER/BigData/Aula1/supermercado (2).csv', sep=';')
pdf.columns = ['produto', 'quantidade', 'preco_unit']
pdf['quantidade'] = pdf['quantidade'].astype(float)
pdf['preco_unit'] = pdf['preco_unit'].astype(float)

# 2) Converte para Dask (2 partições só para ilustrar)
df = dd.from_pandas(pdf, npartitions=2)

# Total por linha e total geral
df = df.assign(total = df['quantidade'] * df['preco_unit'])
valor_total_fim = df['total'].sum()                  # soma lazy (float)

# Contagem de itens (fração conta como 1)
def to_count(q):
    return int(q) if float(q).is_integer() else 1

cont_itens = df['quantidade'].map(to_count, meta=('count', 'i8')).sum()

# Produto mais caro (unitário)
mais_caro = df.nlargest(1, 'preco_unit')[['produto','preco_unit']].compute()

# Execução
vt = valor_total_fim.compute()
n_itens = cont_itens.compute()
mais_caro_row = mais_caro.iloc[0].to_dict() if not mais_caro.empty else {}

# Saída básica
print("\nResumo:")
print(f"Valor total (R$): {vt:.2f}")
print(f"Contagem de itens: {int(n_itens)}")
if mais_caro_row:
    print(f"Mais caro: {mais_caro_row['produto']} (R$ {mais_caro_row['preco_unit']:.2f})")
else:
    print("Mais caro: não encontrado")

# Grafo do map + reduce (sum)
count_series = df['quantidade'].map(to_count, meta=('count','i8'))
try:
    count_series.visualize("dag_map_count.svg", optimize_graph=True)
except Exception:
    print("[Aviso] Pulando visualização do DAG (Graphviz ausente no PATH)")

# 1. Mais barato
mais_barato = df.nsmallest(1, 'preco_unit')[['produto','preco_unit']]

# 2. Médias
media_simples = df['preco_unit'].mean()
media_pond = (df['preco_unit'] * df['quantidade']).sum() / df['quantidade'].sum()

# 3. > R$ 10,00
n_maior_10 = (df['preco_unit'] > 10).sum()

# 4. Categoria por faixa
bins = [-float('inf'), 5, 15, float('inf')]
labels = ['Barato', 'Médio', 'Caro']
cat_meta = pd.Series([], dtype='category', name='categoria')
df = df.assign(
    categoria = df['preco_unit'].map_partitions(
        lambda s: pd.cut(s, bins=bins, labels=labels, include_lowest=True),
        meta=cat_meta
    ))

dist_categorias = df['categoria'].value_counts()

# 5. Top-5 por contribuição
top5 = df.nlargest(5, 'total')[['produto','quantidade','preco_unit','total']]

# Cálculos finais e impressão
mb = mais_barato.compute()
if not mb.empty:
    mb_row = mb.iloc[0].to_dict()
    print(f"Mais barato: {mb_row['produto']} (R$ {mb_row['preco_unit']:.2f})")
else:
    print("Mais barato: não encontrado")

print(f"Média simples (R$): {media_simples.compute():.2f}")
print(f"Média ponderada (R$): {media_pond.compute():.2f}")
print(f"Qtd produtos com preço > R$ 10,00: {int(n_maior_10.compute())}")

dist = dist_categorias.compute().to_dict()
print("Distribuição por categoria:", dist)

print("Top-5 por contribuição:")
print(top5.compute().to_string(index=False))
