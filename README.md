# Tele2 от OaksDungeon
## Описание проекта

Данный проект является вариантом сервиса по автоматическому оформлению торговой точки в стиле Tele2 от команды OaksDungeon

## Ключевые функции

- 🔍 **Обнаружение объектов**: Используя YOLO (You Only Look Once), наша модель способна обнаруживать прилавки, двери, кассы и остальные объекты, на которых могуть ыть размещены тематические материалы с высокой точностью.
  
- 🗒️ **Размещение материалов**: Программа сама располагает соответсвующие материалы (плакаты, наклейки, вывески и тд) в подходящие для соответсвующего объекта места, используя для этого разметку, выполненную YOLO.

- 🫳 **Перемещение и добавление**: Программа позволяет двигать заранее размещенные объекты, а так же добавлять новые.

- 🚮 **Удаление**: Программа позволяет удалить выбранный объект.

- ⬇️ **Скачивание**: Программа позволяет скачать полученное оформление в формате png.

## Как начать

1. Убедитесь, что ваша среда соответствует требованиям из `requirements.txt` и `npm.txt` (так же вам необходим установленный Python3, Uvicorn и npm).
2. Скачайте архивы `assets.7z` и `env.7z`. Распакуйте их по пути `client/`. Должно получится `client/assets` и `client/env`.
3. Скачать архив `node_modules.7z` по ссылке `https://drive.google.com/file/d/1-zNhclfZV6XvZKFn-XcDSI79D0b4eGbU/view?usp=sharing`. Распакуйте его по пути `client/`. Должно получится `client/node_modules`.
4. Запустите виртуальное окружение, запустив `server/.venv/Scripts/activate.bat`
5. Запустите сервер в виртуальном окружение строкой `uvicorn yolo_api:app --host 0.0.0.0 --port 8000`
6. Запустите веб-приложение, перейдя по пути `client/tele2/src` и выполнив команду `npm start`.
7. Откройте браузер и перейдите по адресу `http://localhost:3000` (на устройстве, на котором запущено веб-приложение) или по адресу `http://{ip_адрес_устройства_с_запущенным_веб-приложением}:3000/` (на других устройствах).

## Особенности

- **Масштабируемость**: Наш проект легко улучшить, использовав другие варианты моделей YOLO, что позволит изменить условия выделения объектов.

- **Современный дизайн**: Веб-интерфейс создан с учетом последних трендов в дизайне, обеспечивая вас приятным и продуктивным опытом, при этом сохраняя фирменный стиль сайта Tele2.

- **Открытый исходный код**: Мы приветствуем вклады и обратную связь от сообщества. Присоединяйтесь к нашему проекту и делитесь своим опытом!

## Демонстрация работы программы
### Демонстрация пк-версии



## О команде Oaks Dungeons
### Участники
- Сорокина Александра Валерьевна (капитан)
- Винтерголлер Тимофей Андреевич

### Описание
Команда, состоящая из студентов 2 курса группы 1520321 Института Передовых Информационных Технологий.

**Соединим виртуальное и реальное, создадим будущее вместе!** 🚀🌟

*С любовью, команда Oaks Dungeons*
