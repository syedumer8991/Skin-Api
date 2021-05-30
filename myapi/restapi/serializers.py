from django.db.models import fields
from rest_framework.serializers import Serializer, HyperlinkedModelSerializer,ImageField

from .models import Hero
import base64, uuid
from django.core.files.base import ContentFile

# Custom image field - handles base 64 encoded images
class Base64ImageField(ImageField):
    def to_internal_value(self, data):
        if isinstance(data, str) and data.startswith('data:image'):
            # base64 encoded image - decode
            format, imgstr = data.split(';base64,') # format ~= data:image/X,
            ext = format.split('/')[-1] # guess file extension
            id = uuid.uuid4()
            data = ContentFile(base64.b64decode(imgstr), name = id.urn[9:] + '.' + ext)
        return super(Base64ImageField, self).to_internal_value(data)

class HeroSerializer(HyperlinkedModelSerializer):
    picture = Base64ImageField(max_length=None, use_url=True)
    class Meta:
        model = Hero
        fields = ('id', 'name', 'alias', 'picture')
