#1) data_transformation
 #1.1) resize to (256, 256, 3)
 #1.2) make couple :(image, caption_target)

import torchvision
from torchvision import transforms
data_transformation = torchvision.transforms.Compose([transforms.Resize([224, 224])])

data = torchvision.datasets.CocoCaptions(root = 'val2014/',
                                         annFile = '/content/gdrive/My Drive/captions_val2014.json',
                                         transform = data_transformation,
                                         target_transform = None,
                                         transforms = None)
