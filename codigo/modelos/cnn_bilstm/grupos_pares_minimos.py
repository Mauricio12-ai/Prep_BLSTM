"""Aciertos y confusiones en los grupos candidatos a pares mínimos.

Ejecutar desde la raíz del proyecto:
python codigo/modelos/cnn_bilstm/grupos_pares_minimos.py
python codigo/modelos/cnn_bilstm/grupos_pares_minimos.py --predicciones RUTA/predicciones_test.csv
Sin --predicciones usa el predicciones_test.csv más reciente de la fusión. No entrena ni evalúa nada.
"""
import argparse
import csv
from pathlib import Path

GRUPOS = [['Poor', 'Very_poor'], ['Tall', 'Very_tall', 'Very_short'],
          ['Small_cell_phone', 'Large_cell_phone', 'Extra_large_cell_phone'],
          ['Learn_something', 'Learn_everything'], ['Person', 'People']]

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--predicciones', type=Path)
a = p.parse_args()
ruta = a.predicciones or sorted(Path('resultados/fusion').glob('split_comun_*/test_*/predicciones_test.csv'))[-1]
with ruta.open(encoding='utf-8-sig', newline='') as f:
    filas = list(csv.DictReader(f))

print(f"{'Grupo':<62}{'Muestras':>9}{'Aciertos':>10}{'Interna':>9}{'Externos':>10}")
for grupo in GRUPOS:
    r = [x for x in filas if x['real'] in grupo]
    aciertos = sum(x['real'] == x['predicha'] for x in r)
    interna = sum(x['real'] != x['predicha'] and x['predicha'] in grupo for x in r)
    print(f"{' / '.join(grupo):<62}{len(r):>9}{aciertos:>10}{interna:>9}{len(r) - aciertos - interna:>10}")
print('\nInterna: se predijo otra palabra del mismo grupo. Externos: se predijo una palabra de fuera del grupo.')
print('Archivo leído:', ruta)
