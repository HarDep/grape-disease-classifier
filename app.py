import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import logging
import warnings
import io
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class PredictionResponseModel(BaseModel):
    predicted_class: str
    confidence: float
    all_predictions: dict
    disease_info: dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

class GrapeDiseasePredictor:
    def __init__(self, disease_model_path):
        """
        Inicializar el predictor de enfermedades de uva con validación
        """
        try:
            # Cargar modelo de clasificación de enfermedades
            self.disease_model = tf.keras.models.load_model(disease_model_path)
            logger.info(f"Modelo de enfermedades cargado exitosamente desde {disease_model_path}")
            
        except Exception as e:
            logger.error(f"Error cargando los modelos: {e}")
            raise
            
        self.IMG_SIZE = (128, 128)
        
        # Mapeo de clases para enfermedades
        self.disease_mapping = {
            0: "Black Rot",
            1: "ESCA (Black measles)", 
            2: "Healthy",
            3: "Leaf Blight"
        }
        
        # Descripciones de enfermedades
        self.disease_descriptions = {
            "Black Rot": {
                "emoji": "🔴",
                "description": "Podredumbre negra - Enfermedad fúngica grave que causa manchas circulares marrones en las hojas.",
                "severity": "Alta",
                "treatment": "Fungicidas cúpricos, poda de partes afectadas, mejora de ventilación"
            },
            "ESCA (Black measles)": {
                "emoji": "🟤", 
                "description": "Enfermedad de la ESCA - Complejo de hongos que causa manchas como sarampión en las hojas.",
                "severity": "Muy Alta",
                "treatment": "No hay cura definitiva, manejo preventivo con fungicidas sistémicos"
            },
            "Healthy": {
                "emoji": "🟢",
                "description": "Hoja saludable - Sin signos de enfermedad detectables.",
                "severity": "Ninguna",
                "treatment": "Continuar con prácticas de manejo preventivo"
            },
            "Leaf Blight": {
                "emoji": "🟡",
                "description": "Tizón de la hoja - Enfermedad que causa manchas necróticas y amarillamiento.",
                "severity": "Media",
                "treatment": "Fungicidas preventivos, mejora del drenaje, evitar riego foliar"
            }
        }
        
    def preprocess_image(self, image):
        """
        Preprocesar la imagen para los modelos
        """
        try:
            # Convertir PIL a numpy array si es necesario
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Verificar que la imagen tiene el formato correcto
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("La imagen debe ser en formato RGB")
            
            # Redimensionar a 128x128
            image_resized = cv2.resize(image, self.IMG_SIZE)
            
            # Normalizar (0-1)
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Añadir dimensión de batch
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            return image_batch, image_resized, image_normalized
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            raise
    
    def predict_disease(self, image):
        """
        Realizar predicción de enfermedad en la imagen
        """
        try:
            # Preprocesar imagen
            image_batch, image_resized, image_normalized = self.preprocess_image(image)
            
            # Hacer predicción
            predictions = self.disease_model.predict(image_batch, verbose=0, batch_size=1)
            
            # Obtener clase predicha y confianza
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.disease_mapping[predicted_class_idx]
            
            # Crear diccionario de todas las probabilidades
            all_predictions = {}
            for i, class_name in self.disease_mapping.items():
                all_predictions[class_name] = float(predictions[0][i])
            
            # Obtener información de la enfermedad
            disease_info = self.disease_descriptions.get(predicted_class, {})
            
            return predicted_class, confidence, all_predictions, image_resized, image_normalized, disease_info
            
        except Exception as e:
            logger.error(f"Error en predicción de enfermedad: {e}")
            raise
    
    def predict(self, image):
        """
        Realizar predicción completa: luego clasificar enfermedad
        """
        try:
            
            # Si es hoja de uva, proceder con clasificación de enfermedad
            predicted_class, confidence, all_predictions, _, _, disease_info = self.predict_disease(image)
            
            data = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'disease_info': disease_info
            }

            return PredictionResponseModel(**data)
            
        except Exception as e:
            logger.error(f"Error en predicción completa: {e}")
            raise

app = FastAPI(title="Grape Disease Prediction API", 
              version="1.0", 
              description="API para predecir enfermedades de uva")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

disease_model_path="best_model.h5"

predictor = GrapeDiseasePredictor(disease_model_path)

@app.post("/predict")
async def predict_endpoint(image: UploadFile = File(...)):
    try:
        if image.filename is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No se proporcionó un archivo")
        _, ext = os.path.splitext(image.filename)
        if ext.lower() not in ['jpg', 'jpeg']:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El archivo debe ser una imagen jpg o jpeg")
        image.file.seek(0)
        image_file = image.file.read()
        image_pil = Image.open(io.BytesIO(image_file))
        prediction = predictor.predict(image_pil)
        return prediction
    except Exception as e:
        logger.error(f"Error en la solicitud POST: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error en la solicitud")