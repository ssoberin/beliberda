"""Execute a report notebook from saved metrics only. No training."""
import json,os,sys,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook(cells=[md('''# V3: 250 замен простых текстов

Два обучения с одинаковыми параметрами и порядком: public3398 + v3train3000 + прежние авторские100 =6498 строк. Заменены250 строк v3 на50сюжетов по5версий. Распределения source и меток102–106 сохранены. TF-IDF word(1,3)+char_wb(3,5),18k+28k,SGD,seed42,пороги0.5. Notebook только читает измерения.

250новых строк не являются250независимыми сюжетами. Public729 — validation. Official208 содержит конфликты и пересечения; clean114 — диагностический срез. Проверки уже использовались, public test повлиял на направления дополнения. Независимый финальный тест отсутствует. Авторские метки не проходили независимую человеческую перепроверку.'''),code('''from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display,Markdown,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'avito_compare').exists() and (p/'output').exists())
P=ROOT/'output/v3-revision3-study'
s=json.loads((P/'summary.json').read_text());runs=s['runs']
display(Markdown((P/'report.md').read_text()))
assert s['protocol']['train_rows']==6498
assert runs['before']['fit']['thresholds']==runs['after']['fit']['thresholds']==[.5]*5
'''),code('''keys=['public_validation_729','official_public_test_208','v3_test','v3_challenge','new_control_20']
labels=['Public val729','Public test208','V3 test600','V3 challenge144','Control20']
x=list(range(len(keys)))
fig,ax=plt.subplots(figsize=(10,4))
for offset,name,color in [(-.18,'before','#64748b'),(.18,'after','#0f766e')]:
    bars=ax.bar([i+offset for i in x],[runs[name]['evaluation'][k]['micro_f1'] for k in keys],width=.35,label=name,color=color)
    ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
ax.set_xticks(x,labels);ax.set_ylim(0,1);ax.set_ylabel('Micro-F1');ax.legend();ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(P/'comparison.png',dpi=150);plt.close(fig)
display(Image(filename=str(P/'comparison.png')))
'''),md('''## Решение

Рекомендация: **не заменять исходный v3 этой ревизией как основным train**. Лексика выросла1210→2336словоформ, но V3testF1 снизился0.6818→0.6398: FP107→85, FN425→487. Publictestпочтибезизменений0.8223→0.8206, publicvalidation0.9344→0.9318. Challengeулучшился0.7385→0.7619; этого недостаточно для продвижения. На control20снижение0.4571→0.4000, но это всего10сценариев.

Новые тексты полезны как сохранённый контрастный материал, однако расширение словаря не гарантирует переноса линейного классификатора. Причина ухудшения отдельно не изолирована: в A/B одновременно удалялись исходные и добавлялись новые формулировки. Аддитивное включение и оценка с разбиением по семействам — возможные следующие эксперименты, **не запускались**. Рабочие веса и исходные тесты не изменены.''')])
nb.cells.extend([md("## Дополнительная проверка 46k/60k/100k и порога\nВо всех вариантах word(1,3). Порог выбран только по public calibration728 + v3validation600; последнее теперь калибровочная часть этой серии."), code("display(Markdown((ROOT/'output/v3-revision3-caps/report.md').read_text()))")])
with tempfile.TemporaryDirectory(prefix='v3-r3-nb-') as tmp:
    spec=Path(tmp)/'kernels/revision';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Revision','language':'python'}))
    os.environ.update(JUPYTER_PATH=tmp,IPYTHONDIR=str(Path(tmp)/'ipython'),JUPYTER_RUNTIME_DIR=str(Path(tmp)/'runtime'),MPLCONFIGDIR=str(Path(tmp)/'matplotlib'))
    NotebookClient(nb,kernel_name='revision',timeout=120,resources={'metadata':{'path':str(ROOT)}}).execute()
nb.metadata['kernelspec']={'name':'python3','display_name':'Python3','language':'python'}
nbformat.validate(nb);nbformat.write(nb,P/'comparison.ipynb');print(P/'comparison.ipynb')
