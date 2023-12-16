## Installation guide

```shell
pip install -r ./requirements.txt
```

запуск трейна

```
python hw_scuf/train.py -c hw_asr/hw_asr/configs/final.json
```

После этого надо переложить из папки saved последний чекпоинт и его config в папку `default_test_model`, для запуска теста
переименовать их в `checkpoint.pth` и `config.json` соответственно.

Готовые чекпоинт и конфиг уже лежат в папке `default_test_model`.

запуск теста

```
python hw_scuf/test.py
```
после запуска создается файл `output.json`, в котором записаны вероятности бонафайнд и скуф для всех `7` тестовых записей - 3 преподавательских, 3 сгенерированных и одно реальное из интернета.
