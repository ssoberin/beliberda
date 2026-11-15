"""Render an executed results notebook and figure from frozen metrics only."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/innopolis-catboost-mpl')
os.environ.setdefault('IPYTHONDIR', '/tmp/innopolis-catboost-ipython')
import json
import sys
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient

P = Path('output/catboost-optimization').resolve()
nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nb.cells = [
    nbformat.v4.new_markdown_cell(
        '# CatBoost: 45 обучений, отбор только по calibration\n'
        'Этот notebook визуализирует зафиксированные результаты. Он не обучает модели '
        'и не подбирает параметры. Clean — 152 строки после исключения 56 строк из '
        'конфликтных групп. В public test source_mcId=101 является допущением. '
        'Основная модель не заменена; полная проверка смысла изменений — в report.md '
        'и public_test_semantic_review.md.'),
    nbformat.v4.new_code_cell(
        "import json\nfrom pathlib import Path\nimport numpy as np\nfrom IPython.display import display, HTML\n"
        "import matplotlib.pyplot as plt\n"
        "P = Path.cwd()\nmetrics = json.loads((P/'final_metrics.json').read_text())\n"
        "names = {'C_syntax_baseline__fixed':'C_syntax', "
        "'adaptive_0_3__per_category':'Depth 6 + TF-IDF', "
        "'adaptive_0_2__per_category':'Public-cal finalist', "
        "'adaptive_1_0__fixed':'Depth 8, threshold 0.5'}\n"
        "datasets = ['pub_validation','official_208','public_test_clean','v3_test']\n"
        "f1 = np.array([[metrics[n]['metrics'][d]['micro_f1'] for d in datasets] for n in names])\n"
        "def table(headers, rows):\n display(HTML('<table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'))\n"
        "table(['Model']+datasets, [[name]+[f'{v:.6f}' for v in vals] for name, vals in zip(names.values(),f1)])"),
    nbformat.v4.new_code_cell(
        "fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))\n"
        "x=np.arange(len(datasets)); width=.19; colors=plt.rcParams['axes.prop_cycle'].by_key()['color']\n"
        "for i, name in enumerate(names.values()):\n ax[0].bar(x+(i-1.5)*width, f1[i], width, label=name, color=colors[i])\n"
        "ax[0].set_xticks(x, datasets, rotation=15); ax[0].set_ylim(.78, .99)\n"
        "ax[0].set_ylabel('Micro-F1'); ax[0].set_title('Frozen finalists: observed F1')\n"
        "ax[0].legend(fontsize=8)\n"
        "delta = 100 * (f1[1:] - f1[0])\n"
        "for i, name in enumerate(list(names.values())[1:]):\n ax[1].bar(x+(i-1)*.24, delta[i], .24, label=name, color=colors[i+1])\n"
        "ax[1].set_xticks(x, datasets, rotation=15); ax[1].axhline(0, color='black', linewidth=.7)\n"
        "ax[1].set_ylabel('Change, percentage points'); ax[1].set_title('Change vs C_syntax')\n"
        "ax[1].legend(fontsize=8)\nfig.tight_layout()\n"
        "fig.savefig(P/'comparison.png', dpi=180, bbox_inches='tight')\nplt.show()"),
    nbformat.v4.new_code_cell(
        "rows=[]\nfor n in names:\n for d in datasets:\n"
        "  m=metrics[n]['metrics'][d]; b=metrics['C_syntax_baseline__fixed']['metrics'][d]\n"
        "  rows.append([names[n], d, round(m['micro_precision'],6), "
        "round(m['micro_recall'],6), m['fp'], m['fp']-b['fp'], "
        "m['fn'], m['fn']-b['fn'], round(m['exact_match'],6)])\n"
        "table(['Model','Dataset','P','R','FP','delta FP','FN','delta FN','Exact'],rows)"),
    nbformat.v4.new_markdown_cell(
        'По F1 основной кандидат улучшил три набора, однако V3 FN выросли 80→88. '
        'Групповой 95% CI изменения public test F1: [−0.576; +2.638] п.п.; '
        'он включает ноль. Глубина 8 сохранила порог 0.5 и снизила V3 FN до 64. '
        'Её выигрыш на public test мал и семантически смешанный. '
        'Максимум public test 0.830189 у отдельного финалиста сопровождается '
        'снижением V3 до 0.913767. Это не доказательство универсального улучшения.'),
]
with tempfile.TemporaryDirectory(prefix='catboost-notebook-') as runtime:
    spec = Path(runtime)/'kernels/python3'
    spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Python 3', 'language': 'python',
    }))
    os.environ['JUPYTER_PATH'] = runtime + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    NotebookClient(nb, timeout=120, resources={'metadata': {'path': str(P)}}).execute()
nbformat.write(nb, P/'results.ipynb')
print('Wrote executed results.ipynb and comparison.png')
