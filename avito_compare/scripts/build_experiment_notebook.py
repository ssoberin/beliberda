"""Build and execute a read-only review of the saved experiment measurements."""
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root/'output/experiment-review'
    folder.mkdir(parents=True, exist_ok=True)
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(cells=[
        markdown('''# Эксперименты TF-IDF + SGD: сохранённый прогресс

Этот notebook читает готовые JSON-замеры и строит графики. Он **не обучает модели**, не меняет данные и не публикует веса. Выходы ячеек сохранены для просмотра на GitHub.

Область: категории 102–106, исключение исходного mcId, пороги 0.5. Данные public имеют автоматическую разметку; v2 — синтетический стрессовый корпус. Серверная production-модель здесь не проверялась и не заменялась.

Последовательность: словесные триграммы → char_wb/лимиты словаря → v2 отдельно → public+v2 → парное сравнение старой/смешанной моделей на общих тестах.'''),
        code('''from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'avito_compare').is_dir() and (p/'output').is_dir())
OUT = ROOT/'output/experiment-review'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
def read(path): return json.loads((ROOT/path).read_text())
def table(headers, rows):
    text = '| ' + ' | '.join(headers) + ' |\\n| ' + ' | '.join(['---']*len(headers)) + ' |\\n'
    text += '\\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    display(Markdown(text))
def save(fig, name):
    fig.savefig(OUT/name, bbox_inches='tight')
    plt.close(fig)
    display(Image(filename=str(OUT/name)))
ngrams = read('output/ngram-comparison/comparison.json')
features = read('output/tfidf-feature-experiments/summary.json')
mixed = read('output/mixed-charwb-comparison/evaluation.json')
paired = read('output/paired-data-evaluation/comparison.json')
lengths = read('output/dataset-length-comparison/stats.json')
assert all(m['thresholds'] == [0.5]*5 for m in features['measurements'].values())
print('Данные замеров загружены; обучение не выполнялось.')'''),
        markdown('''## 1. Словесные триграммы

Менялся только word ngram_range: (1,2) → (1,3). Символьный analyzer=char, те же train/calibration/validation и параметры SGD. Прирост Micro-F1 небольшой; bootstrap-интервал включает ноль. Время — один fit, включая TF-IDF и SGD.'''),
        code('''table(['Метрика', 'Биграммы', 'Триграммы', 'Абсолютная Δ', 'Относительная Δ, %'],
      [[r['metric'], f"{r['bigrams']:.6f}", f"{r['trigrams']:.6f}", f"{r['absolute_difference']:+.6f}", f"{r['relative_difference_percent']:+.3f}"] for r in ngrams['comparison']])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, metric, title in zip(axes, ['Micro-F1', 'Training wall time, seconds'], ['Micro-F1 (ось увеличена)', 'Время fit, секунды']):
    r = next(r for r in ngrams['comparison'] if r['metric'] == metric)
    bars = ax.bar(['(1,2)', '(1,3)'], [r['bigrams'], r['trigrams']], color=['#64748b','#0f766e'])
    ax.bar_label(bars, fmt='%.5f' if metric == 'Micro-F1' else '%.2f', padding=4)
    ax.set_title(title)
    if metric == 'Micro-F1': ax.set_ylim(.90,.95)
    else: ax.set_ylim(0,10)
save(fig, '01_word_ngrams.png')'''),
        markdown('''## 2. char_wb и размер словаря на public

Все варианты используют word(1,3), одинаковые пороги и разбиение. sublinear_tf=True уже был исходной настройкой. У вариантов *_linear_tf он выключен отдельно. Лимиты 46k/60k/100k — суммы словесных и символьных признаков.'''),
        code('''names = features['main_experiments']
table(['Вариант','Micro-F1','Macro-F1','P micro','R micro','FP','FN','Fit s','Word/char','Модель MiB'],
      [[n, *[f"{features['measurements'][n][k]:.5f}" for k in ['micro_f1','macro_f1','micro_precision','micro_recall']],
        features['measurements'][n]['confusion_counts']['fp'], features['measurements'][n]['confusion_counts']['fn'],
        f"{features['measurements'][n]['fit_wall_seconds']:.2f}",
        f"{features['measurements'][n]['dimensions']['word']}/{features['measurements'][n]['dimensions']['char']}",
        f"{features['measurements'][n]['artifact_bytes']/1024**2:.2f}"] for n in names])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for analyzer, ids, color in [('char',['baseline','char_60k','char_100k'],'#64748b'),('char_wb',['char_wb','char_wb_60k','char_wb_100k'],'#0f766e')]:
    axes[0].plot([46,60,100], [features['measurements'][n]['micro_f1'] for n in ids], marker='o', label=analyzer, color=color)
    axes[1].plot([46,60,100], [features['measurements'][n]['artifact_bytes']/1024**2 for n in ids], marker='o', label=analyzer, color=color)
axes[0].set(title='Micro-F1 (ось увеличена)', xlabel='Лимит признаков, тыс.', ylim=(.925,.94))
axes[1].set(title='Размер артефакта', xlabel='Лимит признаков, тыс.', ylabel='MiB')
for ax in axes: ax.legend(); ax.grid(alpha=.2)
save(fig, '02_feature_limits.png')'''),
        markdown('''## 3. Обучение только на v2 и на смеси

v2 train содержит 8464 коротких синтетических примера. После min_df получено лишь 6483 word +4209 char признаков: лимиты 46k/60k/100k неактивны, коэффициенты моделей совпали. Для сравнения на смеси использованы 3398 дедуплицированных public train +8464 v2 train; готовые проверочные части не переделывались.

Показываем качество отдельно на источниках: общий смешанный Micro-F1 не отражает отсутствие деградации на real public.'''),
        code('''table(['Смешанная модель', 'Public validation F1 (729)', 'V2 test F1 (3016)', 'Challenge F1 (40)'],
      [[n, *[f"{m[k]['micro_f1']:.6f}" for k in ['public_validation','v2_test','challenge']]] for n,m in mixed['models'].items()])
table(['Корпус','Медиана символов','p95 символов','Медиана слов'],
      [[n,lengths[n]['characters']['median'],lengths[n]['characters']['p95'],lengths[n]['words']['median']]
       for n in ['public_raw','public_dedup_normalized','public_prepared_train','v2_train','v2_test']])
display(Markdown('Синтетика составляет **71.35% документов**, но **21.04% символов** смешанного train. Разница распределений доказана; длина отдельно не доказана как причина изменения F1. См. `output/dataset-length-comparison/lengths.ipynb` и EDA.'))'''),
        markdown('''## 4. Стала ли модель лучше без деградации?

Основная пара имеет одинаковую архитектуру char_wb46k и пороги0.5; отличается только состав обучения. Полный официальный public208 содержит дубли и противоречивые ответы. Дополнительно показана очищенная часть: исключены конфликтующие описания, пересечения со всеми public split и mixed train/calibration, согласованные повторы схлопнуты. Метки не исправлялись.

Очищенная часть меньше и меняет состав проверки, поэтому нужны обе оценки. Эти тесты уже использованы в сравнении; они не являются новым нетронутым финальным набором.'''),
        code('''tests = list(paired['comparisons'])
table(['Набор','Строк','F1 public-only','F1 public+v2','Δ п.п.','FP old→new','FN old→new'],
      [[n, r['public_only']['rows'], f"{r['public_only']['micro_f1']:.6f}", f"{r['public_plus_v2']['micro_f1']:.6f}",
        f"{r['delta_new_minus_old']['micro_f1']*100:+.3f}",
        f"{r['public_only']['fp']}→{r['public_plus_v2']['fp']}", f"{r['public_only']['fn']}→{r['public_plus_v2']['fn']}"]
       for n,r in paired['comparisons'].items()])
labels = ['Public208','Public clean114','Public validation729','V2 test3016','Challenge40']
x = np.arange(len(tests)); width=.36
fig, ax = plt.subplots(figsize=(12,4.5))
for offset, model, title, color in [(-width/2,'public_only','Public only','#64748b'),(width/2,'public_plus_v2','Public + v2','#0f766e')]:
    bars=ax.bar(x+offset,[paired['comparisons'][n][model]['micro_f1'] for n in tests],width,label=title,color=color)
    ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
ax.set(xticks=x,xticklabels=labels,ylabel='Micro-F1',ylim=(0,1.05),title='Одинаковые тесты, одинаковая архитектура, разные обучающие данные')
ax.legend(); ax.grid(axis='y',alpha=.2)
save(fig,'03_old_vs_mixed.png')
display(Markdown('Аудит public208:'))
print(json.dumps(paired['official_audit'],ensure_ascii=False,indent=2))'''),
        code('''fig, axes=plt.subplots(1,2,figsize=(10,3.7))
for ax,name,title in zip(axes,['public_validation_729','v2_test_3016'],['Public validation729','V2 test3016']):
    r=paired['comparisons'][name]
    for offset,model,label,color in [(-.18,'public_only','Public only','#64748b'),(.18,'public_plus_v2','Public+v2','#0f766e')]:
        bars=ax.bar(np.arange(2)+offset,[r[model]['fp'],r[model]['fn']],.36,label=label,color=color)
        ax.bar_label(bars,padding=3)
    ax.set(xticks=[0,1],xticklabels=['FP','FN'],title=title,ylabel='Ошибочные решения по категориям')
    ax.legend(fontsize=8); ax.margins(y=.18)
save(fig,'04_fp_fn.png')'''),
        markdown('''## Вывод и следующий шаг

- `sublinear_tf=True` полезен в выполненных абляциях. Увеличение лимита признаков до60k/100k не улучшило результат; char_wb46k — лучший измеренный кандидат.
- Добавление v2 улучшило его собственный test: Micro-F1 **0.3554→0.4470**, но снизило public validation **0.9371→0.9300**. Challenge не показал однозначного преимущества. Критерий «лучше на сложных случаях без деградации public» пока не выполнен.
- Public raw и synthetic v2 сильно различаются длиной, лексикой и сложностью. Почти весь v2 содержит отрицания; формальная логика не должна подменять обычный поток.
- Далее: подобрать долю/вес синтетики на реальной validation и отдельно разбирать реалистичные ограничения. Новые архитектурные изменения здесь не проверялись. Рабочую модель не заменяли.

## Происхождение и воспроизведение

Все таблицы читаются из сохранённых measurement/summary JSON в output. Там есть SHA-256 источников/разбиений, версии зависимостей, параметры и время fit. Точное обучение и подготовка описаны в `avito_compare/README.md`; для notebook-воспроизведения установите `requirements-experiments.txt` и запустите `python -m avito_compare.scripts.build_experiment_notebook`.

Старые замеры ссылаются на historical commit0b1be5d и старый путь public в mvp; содержимое public соответствует переносу в datasets/public. Коммиты branch/history связывают дальнейшие изменения. Веса, подготовленные JSONL, NPZ и кэши в Git не добавляются; все исходные датасеты уже находятся в main.'''),
    ])
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    for name in ('MPLCONFIGDIR','IPYTHONDIR','JUPYTER_RUNTIME_DIR'):
        directory=Path(tempfile.gettempdir())/('avito-'+name.lower())
        directory.mkdir(exist_ok=True)
        os.environ.setdefault(name,str(directory))
    with tempfile.TemporaryDirectory(prefix='avito-nb-kernel-') as temporary:
        spec=Path(temporary)/'kernels/avito-experiment'
        spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Avito experiment','language':'python'}))
        os.environ['JUPYTER_PATH']=temporary+os.pathsep+os.environ.get('JUPYTER_PATH','')
        NotebookClient(nb,kernel_name='avito-experiment',timeout=120,resources={'metadata':{'path':str(root)}}).execute()
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.validate(nb)
    target=folder/'experiments.ipynb'
    nbformat.write(nb,target)
    print(target)


if __name__=='__main__':
    main()
