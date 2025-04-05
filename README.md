# Brain MRI Classifier

Este projeto utiliza aprendizado de máquina e visão computacional para classificar imagens de ressonância magnética cerebral e detectar a presença de tumores.

## Estrutura
- `src/preprocess.py`: pré-processamento das imagens com OpenCV
- `src/model.py`: definição e treinamento do modelo com Keras
- `main.py`: ponto principal de execução

## Como rodar
```bash
pip install -r requirements.txt
python main.py
```

## Dataset
    https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection