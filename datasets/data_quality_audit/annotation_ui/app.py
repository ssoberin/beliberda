"""Local-only blind annotation UI. Run via `streamlit run app.py --server.address 127.0.0.1`."""
import streamlit as st

from storage import (ANNOTATORS,DATASETS,completeness,export_jsonl,
                     load_answers,save_answer,tasks)

NAMES={102:'Штукатурные работы',103:'Плиточные работы',104:'Сантехника',
       105:'Электрика',106:'Поклейка обоев и малярные работы'}

st.set_page_config(page_title='Слепая разметка услуг',layout='wide')
st.title('Разметка услуг Авито')
st.caption('Работу предлагает сам исполнитель. Выбирайте только явно предлагаемые категории; '
           'для сомнительных случаев оставьте комментарий. Пустой выбор тоже нужно сохранить.')

with st.sidebar:
    dataset=st.selectbox('Набор',list(DATASETS),format_func=lambda x:
        'Public — слепая разметка' if x=='public' else 'Train — повторная проверка')
    annotator=st.selectbox('Разметчик',list(ANNOTATORS),format_func=lambda x:
        'Разметчик 1' if x=='annotator_1' else 'Разметчик 2')
    st.info('Выберите свой профиль и не переключайте его в ходе работы. '
            'Файлы разметчиков независимы; вход защищён только локальным доступом/SSH-туннелем.')

try:
    items=tasks(dataset)
    answers=load_answers(dataset,annotator,items)
except (OSError,ValueError,KeyError) as exc:
    st.error(f'Не удалось открыть задания/ответы: {exc}')
    st.stop()

scope=f'{dataset}:{annotator}'
if st.session_state.get('scope')!=scope:
    st.session_state.scope=scope
    st.session_state.position=0
position=min(st.session_state.get('position',0),len(items)-1)
item=items[position]
item_id=item['item_id']
saved=answers.get(item_id)
stats=completeness(items,answers)

with st.sidebar:
    st.progress(stats['saved']/stats['total'],text=f"Сохранено {stats['saved']} из {stats['total']}")
    if stats['missing_ids']:
        st.caption(f"Осталось {len(stats['missing_ids'])}; пустые ответы учитываются после «Сохранить».")
    else:
        st.success('Все задания заполнены.')
    if st.button('Перейти к первому незаполненному',disabled=not stats['missing_ids']):
        target=stats['missing_ids'][0]
        st.session_state.position=next(i for i,x in enumerate(items) if x['item_id']==target)
        st.rerun()
    st.download_button('Экспортировать сохранённые ответы (JSONL)',
        data=export_jsonl(dataset,annotator,items),
        file_name=f'{dataset}_{annotator}_answers.jsonl',mime='application/x-ndjson',
        disabled=not answers)
    if stats['missing_ids']:
        st.warning('Экспорт пока неполный: отсутствуют '
                   f"{len(stats['missing_ids'])} из {stats['total']} ответов.")
    else:
        st.caption('Экспорт полностью заполнен и пригоден для последующего сопоставления.')

st.subheader(f'Задание {position+1} из {len(items)}')
st.caption(f'ID: {item_id} · исходная категория mcId: {item["source_mcId"]}')
st.text_area('Полный текст объявления',value=item['description'],height=470,
             disabled=True,key=f'text:{scope}:{item_id}')

choice_key=f'labels:{scope}:{item_id}'
comment_key=f'comment:{scope}:{item_id}'
if choice_key not in st.session_state:
    st.session_state[choice_key]=saved['human_labels'] if saved else []
if comment_key not in st.session_state:
    st.session_state[comment_key]=saved['human_comment'] if saved else ''

chosen=st.multiselect('Дополнительные категории (можно выбрать несколько или ни одной)',
    list(NAMES),format_func=lambda x:f'{x} — {NAMES[x]}',key=choice_key)
comment=st.text_area('Комментарий — особенно для неоднозначного случая',
                     key=comment_key,height=120)
saved_labels=saved['human_labels'] if saved else []
saved_comment=saved['human_comment'] if saved else ''
dirty=(sorted(chosen)!=sorted(saved_labels) or comment!=saved_comment)
if saved:
    st.caption('Этот ответ сохранён.' + (' Сейчас есть несохранённые изменения.' if dirty else ''))
elif chosen or comment:
    st.warning('Ответ ещё не сохранён.')
else:
    st.caption('Ответ ещё не сохранён; для пустого ответа нажмите «Сохранить».')

left,mid,right=st.columns(3)
with left:
    if st.button('Предыдущее',disabled=position==0,use_container_width=True):
        if dirty:st.warning('Сначала сохраните изменения.')
        else:
            st.session_state.position=position-1
            st.rerun()
with mid:
    if st.button('Сохранить',type='primary',use_container_width=True):
        try:
            save_answer(dataset,annotator,item,list(chosen),comment,items)
            st.rerun()
        except (OSError,ValueError) as exc:
            st.error(f'Сохранение не удалось: {exc}')
with right:
    if st.button('Следующее',disabled=position==len(items)-1,use_container_width=True):
        if dirty:st.warning('Сначала сохраните изменения.')
        else:
            st.session_state.position=position+1
            st.rerun()
