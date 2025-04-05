import kagglehub
from src.preprocess import load_images_from_folder
from src.model import build_and_train_model
from sklearn.model_selection import train_test_split

# Baixar dataset automaticamente
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

# Usar o caminho retornado pelo kagglehub
X, y = load_images_from_folder(path)

# Continuar normalmente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = build_and_train_model(X_train, y_train, X_test, y_test)
model.save("brain_mri_model.h5")
