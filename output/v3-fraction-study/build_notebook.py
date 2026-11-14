"""Render saved fixed-parameter data-fraction measurements; never train models."""
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    folder=Path(__file__).resolve().parent;root=folder.parents[1]
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook(cells=[
        md('''# Постоянный public train +0/20/60/100% синтетики

В каждом fit сохранены все3398 исходных public train. Синтетический пул:3000 own_v3train +100новых авторских текстов. Добавки:0/620/1860/3100. Подвыборки вложенные, seed42; новые авторские пары сохраняются целиком, темы представлены пропорционально.

Архитектура неизменна: word(1,3),char_wb(3,5),18k+28k,sublinear_tf=True,те же SGD,thresholds0.5. Этот notebook только читает результаты: обучения/подбора параметров нет.

Official public208 — тест организаторов с известными конфликтами/пересечениями. Public729 — **validation**, не финальный test. V3test600 и новые20 сохранялись внеfit. Новые20 —10авторских сюжетов, не репрезентативный независимый человеческий gold.'''),
        code('''from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display,Markdown,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'avito_compare').is_dir() and (p/'output').is_dir())
OUT=ROOT/'output/v3-fraction-study'
s=json.loads((OUT/'summary.json').read_text());runs=s['runs']
def table(headers,rows):
    text='| '+' | '.join(headers)+' |\\n| '+' | '.join(['---']*len(headers))+' |\\n'
    display(Markdown(text+'\\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)))
table(['Добавка, %','Public train','Синтетический train','Official public208 F1','Public validation729 F1','V3test600 F1','Новые20 F1','Fit,s'],[[p,r['public_rows'],r['synthetic_rows'],*[f"{r['evaluation'][k]['micro_f1']:.6f}" for k in ['official_public_test_208','public_validation_729','v3_test_600','new_control_20']],f"{r['fit']['fit_wall_seconds']:.2f}"] for p,r in runs.items()])
assert all(r['public_rows']==3398 and r['fit']['thresholds']==[.5]*5 for r in runs.values())
table(['Добавка,%','Положительных102','103','104','105','106'],[[p,*r['synthetic_positive_counts'].values()] for p,r in runs.items()])'''),
        code('''plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
x=[int(k) for k in runs]
fig,axes=plt.subplots(1,2,figsize=(12,4))
for key,label,color in [('v3_test_600','V3test600','#0f766e'),('new_control_20','Новые20','#d97706'),('v3_challenge_144','V3challenge144','#64748b')]:
    axes[0].plot(x,[r['evaluation'][key]['micro_f1'] for r in runs.values()],marker='o',label=label,color=color)
axes[0].set(title='Отдельные синтетические проверки',ylabel='Micro-F1',ylim=(0,1))
for key,label,color in [('official_public_test_208','Official public208','#0f766e'),('public_validation_729','Public validation729','#64748b')]:
    axes[1].plot(x,[r['evaluation'][key]['micro_f1'] for r in runs.values()],marker='o',label=label,color=color)
axes[1].set(title='Исходные данные (увеличенная шкала)',ylabel='Micro-F1',ylim=(.78,.96))
for ax in axes:ax.set_xticks(x);ax.set_xlabel('% от синтетического train3100');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'01_fraction_quality.png',dpi=150,bbox_inches='tight');plt.close(fig)
display(Image(filename=str(OUT/'01_fraction_quality.png')))'''),
        code('''for name in ['official_public_test_208','official_clean_unique','public_validation_729','v3_test_600','new_control_20','v3_challenge_144']:
    display(Markdown('### '+name))
    table(['Добавка,%','Micro-F1','Macro-F1','P','R','FP','FN','Exact set'],[[p,*[f"{r['evaluation'][name][k]:.5f}" for k in ['micro_f1','macro_f1','micro_precision','micro_recall']],r['evaluation'][name]['fp'],r['evaluation'][name]['fn'],f"{r['evaluation'][name]['exact_set_accuracy']:.5f}"] for p,r in runs.items()])
table(['Добавка,%','Fit,s','PeakRSS,MiB','Модель,MiB','Word/char'],[[p,round(r['fit']['fit_wall_seconds'],3),round(r['fit']['fit_peak_process_rss_mib'],1),round(r['fit']['artifact_bytes']/1024**2,3),f"{r['fit']['dimensions']['word']}/{r['fit']['dimensions']['char']}"] for p,r in runs.items()])'''),
        md('''## Выводы и ограничения

- При добавлении синтетики v3test существенно улучшился; лучший из предзаданных запусков на нём —100%.
- На полном officialpublic208 F1 вырос0.814815→0.822320. Но там есть ошибки разметки и пересечения с исходным train; показываем также очищенные114.
- На исходной validation есть небольшое снижение0.937124→0.934426. Условие полного отсутствия деградации пока не доказано.
- Новые20 показывают иную динамику: лучшийF1 на20%, и лишь1из10контрастныхпар целиком решена смешанными моделями. Нельзя объявлять все сложные случаи закрытыми.
- Никаких альтернативных порогов или гиперпараметров не подбирали. Вывод о количестве синтетики ограничен одной вложенной случайной серией, не причинным эффектом именно100новых строк.
- Ни production-модель, ни исходныйpublic/v3 не заменялись. Версии, настройки, хэши иIDвыбранных примеров:summary.json. Все показатели:report.md. Подробности новых текстов:datasets/context_pairs_100/README.md.

Каноническийv3 snapshot:dataset_v3@0882f80. Его можно получить отдельным gitworktree; обучение описано в avito_compare/README.md. Веса и подготовленныеJSONLвoutput не коммитятся.'''),
    ])
    with tempfile.TemporaryDirectory(prefix='avito-fraction-nb-') as tmp:
        spec=Path(tmp)/'kernels/avito-fractions';spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Avito fractions','language':'python'}))
        os.environ.update({'JUPYTER_PATH':tmp,'IPYTHONDIR':str(Path(tmp)/'ipython'),'JUPYTER_RUNTIME_DIR':str(Path(tmp)/'runtime'),'MPLCONFIGDIR':str(Path(tmp)/'matplotlib')})
        NotebookClient(nb,kernel_name='avito-fractions',timeout=120,resources={'metadata':{'path':str(root)}}).execute()
    nb.metadata['kernelspec']={'name':'python3','display_name':'Python3','language':'python'}
    nbformat.validate(nb);nbformat.write(nb,folder/'fractions.ipynb');print(folder/'fractions.ipynb')


if __name__=='__main__':main()
