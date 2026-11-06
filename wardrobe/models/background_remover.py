from PIL import Image
from rembg import remove, new_session
from typing import Protocol

class BackgroundRemover(Protocol):
    def rem_background(self, image: Image.Image) -> Image.Image:
        ...


class RembgBackgroundRemover:
    def __init__(self,model):
        self.model = model
        self.session = new_session(self.model)
    def rem_background(self,image:Image.Image)-> Image.Image: #remove background of a single image and return the new image
        # Pass the session loaded in __init__. Without it, rembg loads its own default model on every call.
        output = remove(image, session=self.session)
        return output