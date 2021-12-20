# dla_hw4

Для воспроизведения обучения необходимо запустить скрипт для обучения из директории, в которой лежит датасет LJSpeech и репозиторий waveglow. Для скачивания:
```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2

pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r dla_hw4/requirements.txt

pip install gdown

!gdown --id 1t5Hgo1GbcEaZyQrzwQgs8X-h8fcCrFtG
!gdown --id 1IgstYzXNYz4f97rZDSFoR-r9KiDpFEnP
!gdown --id 1oSbmcTxgU-Imx7K__vLr9WP5kV4ubRzb

!gdown --id 1WzlCWYPTUk4UX6iKSPmKQJEaUEGUyffY
!gdown --id 1kl3eEFRK030V9DtWRc4X0wQjw6O_qNw-
!gdown --id 1Gd3KLwl2xRexx97AeOB8kNH6SirDb7FV
```
Запускается обучение и продолжается со скаченных через gdown чекпоинтов, если checkpoint имеется, командой: `python3 dla_hw4/train.py`

Можно воспользоваться ноутбуком `reproduce.ipynb`, который выполняет данный скрипт.
