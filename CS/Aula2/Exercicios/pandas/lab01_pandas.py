import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
file_path = '/Users/akatsurada/Documents/INSPER/CS/Aula2/Exercicios/coffee.csv'
df = pd.read_csv(file_path)

#1. Exibir as primeiras linhas do dataset
print("1A - head():")
print(df.head())

print("\n1B - iloc[:5]:")
print(df.iloc[:5])

#2. Informações gerais sobre os dados
print("\n2A - info():")
df.info()

summary = pd.DataFrame({
    'dtype': df.dtypes,
    'nulls': df.isna().sum(),
    'non_nulls': df.notna().sum(),
    'nunique': df.nunique(dropna=False)
})
print("\n2B - sumário customizado:")
print(summary)

#3. Verificar valores ausentes
print("\n3A - total de nulos por coluna:")
print(df.isnull().sum())

print("\n3B - % de nulos por coluna:")
print((df.isna().mean() * 100).round(2))

#4. Lista de estados únicos
print("\n4A - np.unique:")
print(df['State'].unique())

print("\n4B - drop_duplicates():")
print(df['State'].drop_duplicates().tolist())

#5. Número de cidades únicas
print("\n5A - nunique():", df['City'].nunique())

print("5B - len(unique()):", len(df['City'].unique()))

# 6. Lista de cidades únicas (10 primeiras)
print("\n6A - unique()[:10]:", df['City'].unique()[:10])

print("\n6B - drop_duplicates().head(10):", df['City'].drop_duplicates().head(10).tolist())

print("\n6C - sorted(unique()):", sorted(df['City'].dropna().unique())[:10])

#7. Estatísticas descritivas para colunas numéricas
print("\n7A - describe():")
print(df.describe())

numeric_stats = df.select_dtypes('number').agg(['count', 'mean', 'std', 'min', 'median', 'max']).T
print("\n7B - stats manuais:")
print(numeric_stats)

#outra forma direta usando encadeamento
agg_stats = (df.select_dtypes('number')
             .agg(['count', 'mean', 'median', 'std'])
             .T)
print(agg_stats)

#jeito mais manual
num = df.select_dtypes('number')
extra_stats = num.agg(['mean','std','min','median','max','quantile']).T
extra_stats['cv'] = (extra_stats['std']/extra_stats['mean']).round(2)
extra_stats['iqr'] = (num.quantile(0.75)-num.quantile(0.25))
print(extra_stats[['mean','median','std','cv','iqr']])

#8. Qual é o estado com maior consumo de café?
state_sum = df.groupby('State')['Daily Cups Consumed'].sum()
print("\n8A - idxmax():", state_sum.idxmax())

print("8B - sort_values():", state_sum.sort_values(ascending=False).head(1))

#8c
state_sum.nlargest(5).plot.barh()
plt.title('8C - top x estados por xicara consumida')

#8d
(df.groupby('State', as_index=False)
     .agg(total_cups=('Daily Cups Consumed','sum'))
     .nlargest(1, 'total_cups'))

#9. Quantas pessoas preferem cada tipo de café?
print("\n9A - value_counts():")
print(df['Coffee Type'].value_counts(dropna=False))

print("\n9B - groupby.size():")
print(df.groupby('Coffee Type').size().sort_values(ascending=False))

#9C
pd.crosstab(df['Age Group'], df['Coffee Type'], normalize='index').plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1,1))

#9d
vc = df['Coffee Type'].value_counts(dropna=False)
pd.concat([vc, vc.div(len(df)).mul(100).round(1)], axis=1, keys=['count','pct'])

#10. Qual a média de xícaras de café consumidas por faixa etária?
print("\n10A - groupby.mean():")
print(df.groupby('Age Group')['Daily Cups Consumed'].mean())

print("\n10B - pivot_table:")
print(pd.pivot_table(df, values='Daily Cups Consumed', index='Age Group', aggfunc='mean'))

#10c
(df.groupby('Age Group')['Daily Cups Consumed']
   .mean()
   .pipe(lambda s: s.round(2).sort_values(ascending=False)))

#11. Qual a média de gasto mensal com café por estado?
print("\n11A - groupby.mean():")
print(df.groupby('State')['Monthly Coffee Expense (INR)'].mean())

print("\n11B - pivot_table:")
print(pd.pivot_table(df, values='Monthly Coffee Expense (INR)', index='State', aggfunc='mean'))

#11c
print("\n11C - agg:")
out = (df.groupby('State', as_index=False)
         .agg(avg_spend=('Monthly Coffee Expense (INR)', 'mean'))
         .sort_values('avg_spend', ascending=False, ignore_index=True))
print(out)

#12. Distribuição do número de xícaras consumidas por dia
plt.figure()
df['Daily Cups Consumed'].hist(bins=20)
plt.title('12A - Distribuição de Xícaras por Dia')
plt.xlabel('Xícaras por dia')
plt.ylabel('Frequência')
plt.savefig('12A_distribuicao_xicaras.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
df['Daily Cups Consumed'].plot(kind='hist', bins=20, title='12B - Distribuição de Xícaras por Dia')
plt.xlabel('Xícaras por dia')
plt.ylabel('Frequência')
plt.savefig('12B_distribuicao_xicaras.png', dpi=300, bbox_inches='tight')
plt.close()

#densidade e histograma
plt.figure()
df['Daily Cups Consumed'].hist(bins=20, density=True, alpha=0.35)  
df['Daily Cups Consumed'].plot(kind='density')                   
plt.title('12C - Densidade x Histograma')
plt.xlabel('Xícaras por dia')
plt.ylabel('Densidade')
plt.tight_layout()
plt.savefig('12C_hist_kde.png', dpi=300, bbox_inches='tight')
plt.close()

#13. Qual a marca de café preferida pelos consumidores?
print("\n13A - value_counts:")
print(df['Preferred Coffee Brand'].value_counts(dropna=False))

print("\n13B - groupby.size():")
print(df.groupby('Preferred Coffee Brand').size().sort_values(ascending=False))

#Novas métricas, oq daria p/ achar?
q75 = df['Daily Cups Consumed'].quantile(0.75)
heavy = df[df['Daily Cups Consumed'] >= q75]

print('tomadores de café brabos')
print(heavy['Preferred Coffee Brand'].value_counts(dropna=False))

print(
    heavy['Preferred Coffee Brand']
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(1)
)

print("\n tomadores de café brabos gasto e marca")
print(
    heavy.groupby('Preferred Coffee Brand')['Monthly Coffee Expense (INR)']
         .agg(total_spend='sum', avg_spend='mean')
         .round(2)
         .sort_values('total_spend', ascending=False)
)

heavy_brand_stats = (
    heavy.groupby('Preferred Coffee Brand')
         .agg(
             n_people=('Daily Cups Consumed', 'size'),
             total_cups=('Daily Cups Consumed', 'sum'),
             avg_cups=('Daily Cups Consumed', 'mean'),
             total_spend=('Monthly Coffee Expense (INR)', 'sum'),
             avg_spend=('Monthly Coffee Expense (INR)', 'mean')
         )
)
#Custo médio por xícara dentro do segmento heavy (total_spend / total_cups)
heavy_brand_stats['spend_per_cup'] = (
    heavy_brand_stats['total_spend'] / heavy_brand_stats['total_cups']
)

heavy_brand_stats = (
    heavy_brand_stats
        .sort_values('total_cups', ascending=False)
        .round(2)
)

print("\nResumo dos tomadores brabos - xícaras, gasto e custo por xícara:")
print(heavy_brand_stats)

#Gráfico total_cups vs total_spend 
#Usando n_people, cor = spend_per_cup
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    heavy_brand_stats['total_cups'],
    heavy_brand_stats['total_spend'],
    s=heavy_brand_stats['n_people'] * 10, 
    c=heavy_brand_stats['spend_per_cup'],
    cmap='viridis',
    alpha=0.7,
    edgecolor='k'
)

#Anotar cada ponto com a marca
for brand, row in heavy_brand_stats.iterrows():
    ax.annotate(brand, (row['total_cups'], row['total_spend']), fontsize=8, alpha=0.8)

ax.set_xlabel('Total de xícaras (heavy consumers)')
ax.set_ylabel('Gasto total (INR, heavy consumers)')
ax.set_title('Marcas entre tomadores brabos: volume vs gasto')

cbar = plt.colorbar(scatter)
cbar.set_label('Custo por xícara (INR)')

plt.tight_layout()
plt.savefig('tomadores_brabos_volume_gasto.png', dpi=300, bbox_inches='tight')
plt.close()

#Todos os consumidores por marca
all_brand_stats = (
    df.groupby('Preferred Coffee Brand')
      .agg(
          n_people=('Daily Cups Consumed', 'size'),
          total_cups=('Daily Cups Consumed', 'sum'),
          avg_cups=('Daily Cups Consumed', 'mean'),
          total_spend=('Monthly Coffee Expense (INR)', 'sum'),
          avg_spend=('Monthly Coffee Expense (INR)', 'mean')
      )
)

# Custo médio por xícara no público geral
all_brand_stats['spend_per_cup'] = (
    all_brand_stats['total_spend'] / all_brand_stats['total_cups']
)

all_brand_stats = (
    all_brand_stats
      .sort_values('total_cups', ascending=False)
      .round(2)
)

print("\nResumo geral por marca - xícaras, gasto e custo por xícara:")
print(all_brand_stats)

fig, ax = plt.subplots(figsize=(8, 6))
scatter_all = ax.scatter(
    all_brand_stats['total_cups'],
    all_brand_stats['total_spend'],
    s=all_brand_stats['n_people'] * 10,
    c=all_brand_stats['spend_per_cup'],
    cmap='viridis',
    alpha=0.7,
    edgecolor='k'
)

for brand, row in all_brand_stats.iterrows():
    ax.annotate(brand, (row['total_cups'], row['total_spend']), fontsize=8, alpha=0.8)

ax.set_xlabel('Total de xícaras (todos consumidores)')
ax.set_ylabel('Gasto total (INR, todos consumidores)')
ax.set_title('Marcas entre todos os consumidores: volume vs gasto')

cbar_all = plt.colorbar(scatter_all)
cbar_all.set_label('Custo por xícara (INR)')

plt.tight_layout()
plt.savefig('todas_marcas_volume_gasto.png', dpi=300, bbox_inches='tight')
plt.close()

#Segmentos Light, moderado e heavy
#Preço médio por xícara por linha
with pd.option_context('mode.chained_assignment', None):
    df['spend_per_cup'] = (
        df['Monthly Coffee Expense (INR)'] /
        (df['Daily Cups Consumed'] * 30)
    )

#Segmentos com base nos quartis de consumo diário
q25, q75 = df['Daily Cups Consumed'].quantile([0.25, 0.75])
df['segment'] = pd.cut(
    df['Daily Cups Consumed'],
    bins=[-float('inf'), q25, q75, float('inf')],
    labels=['Light', 'Moderate', 'Heavy'],
    include_lowest=True
)

for seg in ['Light', 'Moderate', 'Heavy']:
    seg_df = df[df['segment'] == seg]
    if seg_df.empty:
        continue  

    seg_stats = (
        seg_df.groupby('Preferred Coffee Brand')
              .agg(
                  total_cups=('Daily Cups Consumed', 'sum'),
                  total_spend=('Monthly Coffee Expense (INR)', 'sum'),
                  n_people=('Daily Cups Consumed', 'size')
              )
    )
    seg_stats['spend_per_cup'] = seg_stats['total_spend'] / seg_stats['total_cups']
    seg_stats = seg_stats.sort_values('total_cups', ascending=False).round(2)

    if seg_stats.empty:
        continue

    #Dividindo pq o tamanho ficou estranho
    max_spend = seg_stats['total_spend'].max()
    size_factor = max_spend / 800  
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_seg = ax.scatter(
        seg_stats['total_cups'],
        seg_stats['total_spend'],
        s=seg_stats['total_spend'] / size_factor,
        c=seg_stats['spend_per_cup'],
        cmap='plasma',
        alpha=0.75,
        edgecolor='k'
    )

    for brand, row in seg_stats.iterrows():
        ax.annotate(brand, (row['total_cups'], row['total_spend']), fontsize=8, alpha=0.8)

    ax.set_xlabel('Total de xícaras')
    ax.set_ylabel('Gasto total (INR)')
    ax.set_title(f'Marcas - Segmento {seg}: preço (cor) vs gasto (tamanho)')

    cbar_seg = plt.colorbar(scatter_seg)
    cbar_seg.set_label('Preço médio por xícara (INR)')

    plt.tight_layout()
    plt.savefig(f'marcas_segmento_{seg.lower()}_bubble.png', dpi=300, bbox_inches='tight')
    plt.close()

#14. As pessoas que visitam cafés frequentemente gastam mais?
print("\n14A - groupby.mean():")
print(df.groupby('Frequency of Café Visits (Per Week)')['Monthly Coffee Expense (INR)'].mean())

print("\n14B - pivot_table:")
print(pd.pivot_table(df, values='Monthly Coffee Expense (INR)', index='Frequency of Café Visits (Per Week)', aggfunc='mean'))

#15. Gerar gráficos histograma (questão livre)
plt.figure()
df['Monthly Coffee Expense (INR)'].hist(bins=25)
plt.title('15A - Gasto Mensal com Café')
plt.xlabel('Gasto (INR)')
plt.ylabel('Frequência')
plt.savefig('15A_gasto_mensal_hist.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
df['Monthly Coffee Expense (INR)'].plot(kind='hist', bins=25, title='15B - Gasto Mensal com Café')
plt.xlabel('Gasto (INR)')
plt.ylabel('Frequência')
plt.savefig('15B_gasto_mensal_hist.png', dpi=300, bbox_inches='tight')
plt.close()

#16. Gerar gráficos scatter (questão livre)
df.plot.scatter(x='Daily Cups Consumed', y='Monthly Coffee Expense (INR)', title='16A - Xícaras vs Gasto')
plt.tight_layout()
plt.savefig('16A_scatter_cups_vs_expense.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
filtered = df[['Daily Cups Consumed', 'Monthly Coffee Expense (INR)']].dropna()
plt.scatter(filtered['Daily Cups Consumed'], filtered['Monthly Coffee Expense (INR)'])
plt.title('16B - Xícaras vs Gasto')
plt.xlabel('Xícaras por dia')
plt.ylabel('Gasto (INR)')

plt.tight_layout()
plt.savefig('16B_scatter_cups_vs_expense.png', dpi=300, bbox_inches='tight')
plt.close()