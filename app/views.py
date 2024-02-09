# from django.shortcuts import render

# # Create your views here.
# import os
# from django.conf import settings
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# from .serializer import ImageUploadSerializer
# from rest_framework import status
# from django.http import JsonResponse
# import joblib  # or appropriate library for loading your model
# import numpy as np
# # from PIL import Image
# from django.core.files.storage import FileSystemStorage 
# import cv2

# # Load the machine learning model
# #model_path = os.path.join(settings.BASE_DIR, 'vgg16_similarity_model.pkl')  # Adjust path as needed
# model = joblib.load('vgg16_similarity_model.pkl')  # Load your model

# # /@api_view(['GET'])
# # def apis(request):
# #     data=


# @api_view(['POST'])
# def predict(request):
#     serializer = ImageUploadSerializer(data=request.data)
#     if serializer.is_valid():
#         # Get the uploaded image
#         uploaded_image = request.FILES['image']
#         uploaded_image2 = request.FILES['image2']

       

#         # Make prediction
#         prediction = model.predict(uploaded_image,uploaded_image2)

#         prediction = model.predict(uploaded_image,uploaded_image2)
#         return Response({'prediction': prediction}, status=status.HTTP_200_OK)
#     else:
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
   








#     # if request.method == 'POST':
#     #     # Deserialize the input images
#     #     serializer = ImageSerializer(data=request.data)
#     #     if serializer.is_valid():
#     #         # Process and preprocess the images
#     #         image1=serializer.validated_data['image1']
#     #         image2=serializer.validated_data['image2']
#     #         pre=model.predict(image1,image2)
#     #         # Return the predictions as JSON response
#     #         return JsonResponse({'predictions': pre})
#     #     else:
#     #         return JsonResponse(serializer.errors, status=400)

from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializer import ImageUploadSerializer
from rest_framework import status
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.conf import settings 
from tensorflow.python.keras.backend import set_session
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from keras.applications import vgg16
import joblib

@api_view(['POST'])
def predict(request):
    # Load your model
    model = joblib.load('vgg16_similarity_model.pkl')
    
    serializer = ImageUploadSerializer(data=request.data)
    if serializer.is_valid():
        # Get the uploaded images from the serializer
        image = serializer.validated_data['image']
        image2 = serializer.validated_data['image2']
        
        # Save the uploaded images using Django's default storage system
        file_name1 = "image.jpg"
        file_name2 = "image2.jpg"
        file_name_1 = default_storage.save(file_name1, image)
        file_name_2 = default_storage.save(file_name2, image2)
        file_url1 = default_storage.url(file_name_1)
        file_url2 = default_storage.url(file_name_2)
        
        # Load and preprocess the first image
        original1 = load_img(file_url1, target_size=(224, 224))
        numpy_image1 = img_to_array(original1)
        image_batch1 = np.expand_dims(numpy_image1, axis=0)
        processed_image1 = vgg16.preprocess_input(image_batch1.copy())
        
        # Load and preprocess the second image
        original2 = load_img(file_url2, target_size=(224, 224))
        numpy_image2 = img_to_array(original2)
        image_batch2 = np.expand_dims(numpy_image2, axis=0)
        processed_image2 = vgg16.preprocess_input(image_batch2.copy())
        
        # Make predictions for both images
        with settings.GRAPH1.as_default():
            set_session(settings.SESS)
            predictions1 = model.predict(processed_image1)
            predictions2 = model.predict(processed_image2)
        
        # Decode predictions for both images
        label1 = decode_predictions(predictions1)
        label1 = list(label1)[0]
        label2 = decode_predictions(predictions2)
        label2 = list(label2)[0]
        
        response = {
            'image1_label': str(label1),
            'image2_label': str(label2)
        }
        return JsonResponse(response)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
