import pandas as pd
import numpy as np
import re
from functools import reduce
from operator import add

# ==== Leitura e normalização de nomes (evita bugs 'Cards' vs 'cards') ====
df = pd.read_csv("Credit.csv")

# normaliza nomes de colunas p/ minúsculas e tira espaços
df.columns = [c.strip().lower() for c in df.columns]

# ==== A. Soma do saldo (Balance) via reduce() ====
# 1) garante numérico
balance_numeric = df['balance'].map(lambda v: pd.to_numeric(v, errors='coerce'))
# 2) trata NaN como 0 e usa reduce com acumulador inicial 0.0
balance_sum_reduce = reduce(lambda acc, v: acc + (v if pd.notna(v) else 0.0), balance_numeric, 0.0)

print("A) Soma via reduce:", balance_sum_reduce)
# (Opcional) checagem contra o alvo esperado 208006 (se o dataset for exatamente o do enunciado):
# print("Confere com 208006?", abs(balance_sum_reduce - 208006) < 1e-9)

# mostra situação atual
print("\nHEAD após A)")
print(df.head(5))


# ==== (Opcional) B. Padronização de Gender, idempotente e robusta ====
# Só se a coluna existir
if 'gender' in df.columns:
    df['gender_2'] = df['gender'].apply(
        lambda x: ("M" if str(x).strip().lower() in {"male","m"} 
                   else "F" if str(x).strip().lower() in {"female","f"} 
                   else np.nan)
    )
    print("\nHEAD após B) (gender_2)")
    print(df[['gender','gender_2']].head(5))


# ==== C. Student -> binário (Yes->1 / No->0) com apply/lambda ====
# aceita variações ('yes','y','1', 'no','n','0')
df['student_bin'] = df['student'].apply(
    lambda s: 1 if str(s).strip().lower() in {"yes","y","1","true","t"} 
    else 0 if str(s).strip().lower() in {"no","n","0","false","f"} 
    else np.nan
)

print("\nHEAD após C) (student_bin)")
print(df[['student','student_bin']].head(5))


# ==== D. Função score() e coluna Score via map() ====
# Fórmula pedida: ((cards + age) ^ student) / education
# ATENÇÃO: em Python, potência é ** (não ^). Implementaremos exatamente a fórmula matemática:
# ((cards + age) ** student) / education
def score(cards, age, student, education):
    # coerções numéricas seguras
    c = pd.to_numeric(cards, errors='coerce')
    a = pd.to_numeric(age, errors='coerce')
    s = pd.to_numeric(student, errors='coerce')
    e = pd.to_numeric(education, errors='coerce')
    # validação
    if any(pd.isna(v) for v in (c, a, s, e)) or e == 0:
        return np.nan
    return ((c + a) ** s) / e

# garante colunas numéricas base com map (estilo funcional)
df['cards']     = df['cards'].map(lambda v: pd.to_numeric(v, errors='coerce'))
df['age']       = df['age'].map(lambda v: pd.to_numeric(v, errors='coerce'))
df['education'] = df['education'].map(lambda v: pd.to_numeric(v, errors='coerce'))

# cria Score usando map() (sem apply(axis=1))
df['score'] = list(map(score, df['cards'], df['age'], df['student_bin'], df['education']))

print("\nHEAD após D) (score)")
print(df[['cards','age','student_bin','education','score']].head(5))


# ==== E. Income -> float com lambda 'convert_income' + map() ====
# Robustez: remove $, vírgulas, espaços, e trata strings vazias/None
convert_income = lambda v: (
    float(re.sub(r'[^\d\.\-]', '', str(v)))  # só dígitos, ponto e sinal
    if (v is not None) and re.search(r'\d', str(v)) 
    else np.nan
)
df['income_float'] = list(map(convert_income, df['income']))

print("\nHEAD após E) (income_float)")
print(df[['income','income_float']].head(5))


# ==== F. Existe alguém com renda > 300k? com any() ====
tem_renda_muito_alta = any(map(lambda v: (pd.notna(v) and v > 300_000), df['income_float']))
print("\nF) Algum usuário com renda > 300k?:", tem_renda_muito_alta)


# ==== G. Pessoas com mais de 5 cartões (filter()) ====
# Usamos filter sobre registros (dicts) e recombinamos num DataFrame
registros = df.to_dict('records')
mais_de_5 = list(filter(lambda row: (
    ('cards' in row) and (row['cards'] is not None) and not pd.isna(row['cards']) and (float(row['cards']) > 5)
), registros))

df_mais_de_5 = pd.DataFrame(mais_de_5) if mais_de_5 else pd.DataFrame(columns=df.columns)

print("\nG) HEAD das pessoas com > 5 cartões")
print(df_mais_de_5.head(5))
