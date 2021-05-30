from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet, ViewSet 
from rest_framework.response import Response

from .serializers import HeroSerializer
from .models import Hero

# Create your views here.

class HeroViewSet(ModelViewSet):
    queryset = Hero.objects.all().order_by('name')
    serializer_class = HeroSerializer

# class UploadViewSet(ViewSet):
#     serializer_class = UploadSerializer

#     def list(self, request):
#         return Response('GET API')
    
#     def create(self, request):
#         file_uploaded = request.FILES.get('file_uploaded')
#         content_type = file_uploaded.content_type
#         response = "POST API and you have uploaded a {} file".format(content_type)
#         return Response(response)
