"""Hand-authored adversarial checks; run after vLLM is ready."""
import argparse
from scripts.qwen_classifier import classify

CASES = [
 ('Штукатурим стены, укладываем плитку. Электрику не делаем.', [102,103]),
 ('Делаем ремонт квартир под ключ.', []),
 ('Заменим розетки и выключатели, смонтируем электрощит.', [105]),
 ('Сантехнику выполняет другой мастер, мы укладываем плитку.', [103]),
 ('Сами не клеим обои, можем посоветовать специалиста.', []),
 ('Шпаклюем стены под покраску, красим потолки.', [106]),
 ('Штукатурим стены по маякам и механизированным способом.', [102]),
 ('Укладываем керамогранит, затираем швы, облицовываем фартук.', [103]),
 ('Подключаем стиральные машины к водоснабжению и сливу.', [104]),
 ('Проводим ремонт ванных под ключ без описания отдельных этапов.', []),
 ('При необходимости заказчик сам приглашает электрика.', []),
 ('Установим электрический тёплый пол и заменим проводку.', [105]),
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    a=p.parse_args()
    ok=0
    for i,(desc,gold) in enumerate(CASES,1):
        actual=classify(desc,101,base_url=a.base_url)['mcIds']
        status='OK' if actual==gold else 'FAIL'
        ok += actual==gold
        print(f'{i:02d} {status}: expected={gold} actual={actual} | {desc}',flush=True)
    print(f'Passed {ok}/{len(CASES)}')

if __name__=='__main__':
    main()
