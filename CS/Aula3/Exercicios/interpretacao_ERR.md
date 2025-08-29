# ERR (Predito/Esperado) — Estado e Medida

## Pressupostos
- Para cada **Estado × Medida**, calculamos a relação **Predito/Esperado (ERR)** **ponderada pelo número de altas**.
- **Cores**: branco ~ **1,0** (esperado), vermelho **> 1,0** (pior), azul **< 1,0** (melhor).
- **Medidas**: Infarto (AMI), Doença pulmonar (DPOC/COPD), Insuficiência cardíaca (HF), Quadril/Joelho, Pneumonia (PN).
- Seleciona os **20** estados com mais altas e centraliza o mapa em **1,0**.

## Interpretação 
- **Quadril/Joelho**. Melhor que o esperado: vários estados aparecem em azul (ND, VT, IN, NY, IA, NH, MD, TN). Apesar de ter 10 azuis e 10 vermelhos, os azuis são mais intensos e os vermelhos menos. 
- **Doença pulmonar** e **Pneumonia**. Abaixo do esperado: muitos estados em vermelho; são focos claros de pior desempenho. Pulmonar (6 cinzas x 14 vermelhos, com vermelhos intensos) e pneumonia (3 azuis claros e 2 cinzas x 15 vermelhos) 
 **Insuficiência cardíaca** e **Infarto**. Misto com predominância abaixo do esperado: variação relevante entre estados, mas mais vermelhos claros no geral.
### Artefatos
- Matriz usada no gráfico: `/Users/akatsurada/Documents/INSPER/CS/Aula3/Exercicios/err_state_measure.csv`
- Figura do heatmap: `/Users/akatsurada/Documents/INSPER/CS/Aula3/Exercicios/err_heatmap.png`
