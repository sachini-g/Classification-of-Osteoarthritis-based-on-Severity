# Osteoarthritis Severity Detection

This project is a **deep learning-based web application** that predicts the severity of osteoarthritis (OA) from knee X-ray images using **transfer learning**.  

The app classifies X-rays into five categories (KL grades):  
- 0: No Osteoarthritis  
- 1: Doubtful OA  
- 2: Mild OA  
- 3: Moderate OA  
- 4: Severe OA  

## Key Features

- **Transfer Learning**: Built on a pre-trained **MobileNetV3** model for efficient feature extraction.  
- **FastAPI Web App**: Users can upload an X-ray image and get instant predictions.  
- **User-Friendly Frontend**: Clean interface with clear severity and confidence scores.  
- **All Probabilities Returned**: For transparency and debugging.  

## Tech Stack

- Python, TensorFlow/Keras  
- FastAPI + Uvicorn  
- HTML/CSS for frontend  
