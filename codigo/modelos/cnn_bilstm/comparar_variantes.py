"""Tabla comparativa de test de las tres variantes (landmarks, fusión, RGB).

Ejecutar desde la raíz del proyecto:
python codigo/modelos/cnn_bilstm/comparar_variantes.py
Lee el metricas_test.json más reciente de cada variante; no entrena ni evalúa nada.
"""
import json
from pathlib import Path

VARIANTES = [('Solo landmarks', 'bilstm'), ('Fusión', 'fusion'), ('Solo RGB', 'cnn_bilstm')]

filas = []
for nombre, carpeta in VARIANTES:
    pruebas = sorted(Path('resultados', carpeta).glob('split_comun_*/test_*/metricas_test.json'))
    if not pruebas:
        raise SystemExit(f'No hay metricas_test.json en resultados/{carpeta}')
    m = json.loads(pruebas[-1].read_text(encoding='utf-8'))
    sujetos = {r['sujeto']: r['accuracy'] for r in m.get('por_sujeto', [])}
    filas.append([nombre, f"{m['correct']}/{m['samples']}", f"{m['accuracy']:.2%}", f"{m['macro_f1_34']:.3f}",
                  f"{sujetos['SUBJECT_5']:.1%}" if 'SUBJECT_5' in sujetos else '-',
                  f"{sujetos['SUBJECT_14']:.1%}" if 'SUBJECT_14' in sujetos else '-', str(pruebas[-1].parent)])

titulos = ['Variante', 'Aciertos', 'Exactitud', 'F1-macro', 'Sujeto 5', 'Sujeto 14']
anchos = [max(len(x) for x in col) for col in zip(titulos, *[f[:6] for f in filas])]
print('  '.join(t.ljust(a) for t, a in zip(titulos, anchos)))
print('  '.join('-' * a for a in anchos))
for f in filas:
    print('  '.join(x.ljust(a) for x, a in zip(f[:6], anchos)))
print('\nArchivos leídos:')
for f in filas:
    print(' ', f[0] + ':', f[6])
