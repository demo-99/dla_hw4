# dla_hw4

Для воспроизведения обучения необходимо запустить скрипт для обучения из директории, в которой лежит датасет LJSpeech и репозиторий waveglow. Для скачивания:
```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2

pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r dla_hw4/requirements.txt

pip install gdown
gdown --id 
```
Запускается обучение и продолжается со скаченных через gdown чекпоинтов, если checkpoint имеется, командой: `python3 dla_hw4/train.py`

Можно воспользоваться ноутбуком `reproduce.ipynb`, который выполняет данный скрипт.
