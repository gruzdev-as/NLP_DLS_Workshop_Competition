# NLP_DLS_Workshop (1st Place)

**Решение задачи множественной классификации обратной связи от пользователей**

Структура проекта: 
- В папке [images](images) - Вспомогательные картинки для README.md
- В папке [notebooks](notebooks) - ноутбуки, по которым тренировались модели.
- В папке [data](data) - некоторые данные, необходимые для запуска "псевдоинференса" (Подробнее в пункте про Streamlit)
- В основной папке проекта - файлы для запуска streamlit-based обертки для модели

- Скор на паблик лидерборде: ~ 0,603 accuracy по полному совпадению меток
- Скор на приват лидерборде: ~ 0.607 (1st Place)

Адаптеры на *HuggingFace*: 
- [Gemma-9b](https://huggingface.co/TheStrangerOne/gemma-2-9b-it-bnb-4bit-lora-multilabel)
- [Gemma-27b](https://huggingface.co/TheStrangerOne/gemma-2-27b-it-bnb-4bit-lora-multilabel)

*Если я забыл их открыть и сделать публичными, пожалуйста, свяжитесь со мной...*

Референсные ноутбуки: 
- [Мультилейбл классификация используя Мистраль](https://huggingface.co/blog/sirluk/multilabel-llm)
- [Тренируем Гемму-9б с ЛоРА адаптером](https://www.kaggle.com/code/emiz6413/training-gemma-2-9b-4-bit-qlora-fine-tuning#What-is-QLoRA-fine-tuning?)
- [Настройки адаптера для 27б модели](https://huggingface.co/ethedeltae/new-gemma2-27b-lora_model/blob/main/adapter_config.json)

P.S.

Если где-то в коде остались API ключи, то они не валидны ;) 

## Описание решения. 
### Воспроизведение.

XML-Roberta, описанная в файле [xml_Roberta_train.ipynb](notebooks/xml_Roberta_train.ipynb) выдавала 0 по всем метрикам, что было принято как личное оскорбление. В результате анализа первой страницы гугла была найдена следующая статья: [Мультилейбл классификация используя Мистраль](https://huggingface.co/blog/sirluk/multilabel-llm)

Было решено попробовать тренировать Мистраль, как это описано в файле [Mistral-7B_train.ipynb](notebooks/Mistral-7B_train.ipynb), что позволило в пике выбить точность порядка ~0,5. Однако, шейкапы на лидерборде быстро показали, что это не предел.

Дальнейший поиск привел на кагл к ноутбуку [тренируем Гемму-9б с ЛоРА адаптером](https://www.kaggle.com/code/emiz6413/training-gemma-2-9b-4-bit-qlora-fine-tuning#What-is-QLoRA-fine-tuning?). Gemma-9b позволила получить пиковую точность ~0,56 при трех эпохах. Тренировка описана в файле [notebooks/Gemma-9b_train.ipynb](Gemma-2-9b_train.ipynb). Веса адаптера можно взять соответственно [Здесь](https://huggingface.co/TheStrangerOne/gemma-2-9b-it-bnb-4bit-lora-multilabel)

В качестве финального аккорда, было решено, что стрелять из пушки по голубям - это скучно, давайте выстрелим из ядерной бомбы. Тренировка модели на 27 млдр. параметров gemma-2-27b описана в файле [Gemma-2-27b_train.ipynb](notebooks/Gemma-2-27b_train.ipynb). Веса адаптера можно взять соответственно [Здесь](https://huggingface.co/TheStrangerOne/gemma-2-27b-it-bnb-4bit-lora-multilabel).

### Инференс и генерация сабмишна

Чтобы избежать потенциальных ошибок, инференс и тренировка находятся в одном ноутбуке. Проанилизировав исходную информацию и описания трендов был выдвинут следующий тезис: "В датасете не может быть наблюдений, в которой не определено метки класса. В качестве базового трешхолда для классификации наилучший скор на кросс-валидации показал 0.55. Таким образом, метки в финальном решении после предсказания проставляются по следующей логике: 
1) Классифицируем для каждого наблюдения с трешхолдом 0,55 
2) Для каждой строки, для которой не определилось ни одного класса, понижаем трешхолд для классификации, выбирая его, соответственно последовательно из списка [0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2] (значение экспериментальные и выбраны эмпирически. Можно подобрать более точно. Optuna?)
3) Для семплов, оставшихся без меток проставляется класс 19 - Нет смысла. 

Рекомендую запускать тренировку моделей через Run&Commit на каггле используя 2xT4 GPU. Тренировка занимает: 
- Примерно 8 часов для трех эпох Мистрали
- Примерно 4 часа на две эпохи Геммы 9б
- Примерно 10 часов для одной эпохи Геммы 27b

Снизить время тренировки потенциально можно увеличив число шагов до сохранения чекпоинт стейта модели. 

## Streamlit (дополнительное задание)

В качестве дополнительного задания с использованием фреймворка Streamlit написан "продуктовый" интерфейс для продукта. Критерии выполненного задания: 
- Выполнена работоспособная реализация задания с использованием Streamlit со всеми работающими функциями (2 балла);
- Вывод вероятность принадлежности отрывка текста к каждому из классов (1 балл);
- Наличие валидации при вводе текста (1 балл);
- Наличие Docker-контейнера (1 балл);
- Графическое представление принадлежности текста к классам (1 балл).

### Установка зависимостей

Склонируйте репозиторий 

```bash
git clone https://github.com/gruzdev-as/NLP_DLS_Workshop.git 
```

Создайте виртуальное окружение и активируйте его. Для Windows:
```bash
 python -m venv venv && venv\Scripts\activate
```

Для UNIX-based: 
```bash 
python -m venv venv && source venv/bin/activate
```

Установите зависимости
```bash 
pip install -r requirements.txt
```

Запустите локальный сервер. Браузер со страницей откроется автоматически
```bash
streamlit run application.py
``` 

### Описание работы 

В виду того, что локально у меня не было возможности запустить модель на GPU, было решено внедрить небольшой "грязный хак" в виде чекбокса "pseudo_inference". При включении этой опции, программа не будет подгружать модели, а использует заранее сделанные предсказания из папки **/data**, выбрав один случайных из них. Процесс инференса в этом случае будет выглядеть таким образом: 

<p float="center">
<img src="/images/1.gif" width="30%" height="30%"/>
<img src="/images/2.gif" width="30%" height="30%"/>
<img src="/images/3.gif" width="30%" height="30%"/>
</p>

Пользователю интерфейса выдается: 
- График вероятности для каждого класса 
- Предсказанные классы
- Оценка, теги и текст отзыва 

Чтобы запустить "честный" инференс, не трогайте чекбокс, введите путь к репозиториям адаптера и модели (по-умолчанию всё настроено для самой лучшей модели - Геммы 27б) и нажмите кнопку *Load Models*. После инициализации моделей, вам откроется интерфейс введения обратной связи от пользователя. После заполнения всех полей (проверяется, что поле ввода информации не пустое. Максимальная длина отзыва - 250 символов), информация объединится и прийдет на вход модели. дальнейший алгоритм работы идентичен описанному выше для "псевдоинференса"

<p float="center">
<img src="/images/4.gif" width="100%" height="100%"/>
</p>

## Запуск в докер-контейнере

Возможен запуск разработанного решения с его стримлит оболочкой внутри докер-контейнера. Для этого необходимо выполнить следующие шаги:

1) Соберите образ, используя команду 
```bash 
docker build -t nlp_dls_workshop .
```

2) Запустите контейнер (с поддержкой ГПУ инференса), используя команду
```bash 
docker run -it --gpus all -p 8501:8501 nlp_dls_workshop
```

3) Откройте браузер и перейдите по адресу 
```
http://localhost:8501
```

## Что пробовал, что не пробовал, но хотелось бы попробовать

Пробовал, но не дало результатов: 
- Псевдолейблинг для Gemma-2-9b из теста семплов с уверенностью выше 0.95 не дал прироста на кросс-валидации и ЛБ 
- Блендинг моделей Мистраль и Гемма-9б не дал прироста на кросс-валидации и ЛБ

Не пробовал, но могло бы помочь: 
- Уделить больше времени на анализ текста, попарсить его регулярками. 
- Синтетически расширить датасет за счет аугментаций по трендам используя классические алгоритмы или другую сеть.
- Дистиллировать модель во что-то более легковесное, чтобы продакшн не упал в обморок от гениальности решения. 
