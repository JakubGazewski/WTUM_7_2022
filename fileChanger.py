from time import sleep
from PIL import Image
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import numpy as np

def mmm():
    file = "C:\\\\Users\\Paweu\\Downloads\\the-nature-conservancy-fisheries-monitoring\\train\ALB\\img_00043.jpg"

    img_pil = Image.open(file)
    
    print("A")

    doNothing = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(), 
    ])
    doNothing_image = doNothing(img_pil)

    _transforms = transforms.Compose([
            transforms.RandomEqualize(1),
            transforms.Resize((256, 256)), 
            transforms.ToTensor(), 

            ])
    print("B")

    
    normalized_img = _transforms(img_pil)

    img_an = np.array(normalized_img)
    img_np = np.array(doNothing_image)

    plt.imshow(img_np.transpose(1, 2, 0))
    plt.show()
    plt.imshow(img_an.transpose(1, 2, 0))
    
    #plt.hist(img_np.ravel(), bins=50, density=True)
    #plt.hist(img_an.ravel(), bins=50, density=True)
    plt.show()

mmm()
