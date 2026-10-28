"""Muestra las matrices guardadas en metricas.json sin entrenar nuevamente."""
import argparse
import html
import json
from pathlib import Path

CLASES = ['Analyze','Better','Complaint','Crowd','Cunning','Decrease','Disappear','Envy','Extra_large_cell_phone','Fail','Fine','Forget','Friday','Green','Language','Large_cell_phone','Learn_everything','Learn_something','Live','People','Person','Poor','Progress','Punish','See','Shout','Small','Small_cell_phone','Stagnate','Tall','Tongue','Very_poor','Very_short','Very_tall']

def exportar(report, salida, clases=None):
    """Exporta una tabla HTML por partición; filas=reales, columnas=predichas."""
    clases = CLASES if clases is None else clases
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    archivos = []
    for clave, nombre in [('best_validation', 'validacion'), ('test', 'test')]:
        if clave not in report:
            continue
        cm = report[clave]['confusion']
        n = len(clases)
        if len(cm) != n or any(len(row) != n for row in cm):
            raise ValueError('La matriz y el número de clases no coinciden')
        if any(not isinstance(v, int) or v < 0 for row in cm for v in row):
            raise ValueError('La matriz debe contener conteos enteros no negativos')
        total = sum(map(sum, cm))
        correctos = sum(cm[i][i] for i in range(n))
        cabecera = '<tr><th class="esquina">Real ↓ / Predicción →</th>' + ''.join('<th scope="col">'+html.escape(c)+'</th>' for c in clases) + '<th>Total real</th></tr>'
        filas = []
        for i, row in enumerate(cm):
            celdas = []
            for j, valor in enumerate(row):
                estilo = 'acierto' if i == j else ('error' if valor else 'cero')
                titulo = html.escape(f'Real: {clases[i]} | Predicción: {clases[j]} | Muestras: {valor}', quote=True)
                celdas.append(f'<td class="{estilo}" title="{titulo}">{valor}</td>')
            filas.append('<tr><th scope="row">'+html.escape(clases[i])+'</th>'+''.join(celdas)+f'<td class="total">{sum(row)}</td></tr>')
        filas.append('<tr><th>Total predicho</th>'+''.join(f'<td class="total">{sum(row[j] for row in cm)}</td>' for j in range(n))+f'<td class="total">{total}</td></tr>')
        accuracy = f'{correctos/total:.2%}' if total else 'sin muestras'
        documento = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>Matriz de confusión</title>
<style>body{font-family:Arial,sans-serif;margin:24px;color:#172b43}h1{font-size:24px}.tabla{overflow:auto;max-height:76vh;border:1px solid #aab7c4}table{border-collapse:separate;border-spacing:0;font-size:12px}th,td{padding:9px;text-align:center;border-right:1px solid #d5dce3;border-bottom:1px solid #d5dce3;min-width:38px}thead th{position:sticky;top:0;background:#e7eef7;z-index:2;writing-mode:vertical-rl;transform:none;height:165px}tbody th{position:sticky;left:0;text-align:left;white-space:nowrap;background:#e7eef7;z-index:1}.esquina{writing-mode:horizontal-tb;left:0;z-index:3;min-width:170px}.acierto{background:#bce7d0;font-weight:bold}.error{background:#ffd7ad;font-weight:bold}.cero{color:#a0aab5}.total{background:#eff2f6;font-weight:bold}p{line-height:1.5}</style>'''
        documento += f'<h1>BiLSTM · Matriz de confusión de {nombre}</h1><p><b>{n} filas × {n} columnas de clases.</b> Filas: seña real. Columnas: seña predicha.<br>Verde (diagonal): aciertos. Naranja: confusiones. Los valores son cantidades de muestras, no porcentajes.<br><b>{correctos} aciertos de {total} muestras · Accuracy: {accuracy}</b></p><p>Desplázate horizontal y verticalmente para ver toda la matriz. Pasa el cursor sobre una celda para ver sus dos etiquetas.</p>'
        documento += '<div class="tabla"><table><thead>'+cabecera+'</thead><tbody>'+''.join(filas)+'</tbody></table></div></html>'
        archivo = salida / f'matriz_confusion_{nombre}.html'
        archivo.write_text(documento, encoding='utf-8')
        archivos.append(archivo)
        print('Matriz guardada:', archivo.resolve())
    if not archivos:
        raise ValueError('No se encontró best_validation ni test en las métricas')
    return archivos

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metricas', type=Path, required=True)
    parser.add_argument('--salida', type=Path)
    args = parser.parse_args()
    report = json.loads(args.metricas.read_text(encoding='utf-8-sig'))
    exportar(report, args.salida or args.metricas.parent)

if __name__ == '__main__':
    main()
