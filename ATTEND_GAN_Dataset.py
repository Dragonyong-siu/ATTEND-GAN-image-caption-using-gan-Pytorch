#3) ATTEND_GAN_Dataset
 #3.1) ATTEND_GAN_Dataset_LSTM
  # feature_map from pretrained ResNet-152
  # using gpt2_tokenizer & gpt2_encoder
  # feature_map
  # caption_ids
  # caption_target

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model

special_tokens = '[pad]', '[start]', '[end]'
GPT2_Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
GPT2_Config = GPT2Config.from_pretrained('gpt2')
GPT2_Model = GPT2Model(GPT2_Config).from_pretrained('gpt2', config = GPT2_Config)
GPT2_Tokenizer.add_tokens(special_tokens)
GPT2_Model.resize_token_embeddings(len(GPT2_Tokenizer))

class ATTEND_GAN_Dataset(torch.utils.data.Dataset):
  def __init__(self, data, max_len, feature_extractor, tokenizer, target_num):
    self.data = data
    self.max_len = max_len
    self.feature_extractor = feature_extractor
    self.tokenizer = tokenizer
    self.encoded_pad = self.tokenizer.convert_tokens_to_ids('[pad]')
    self.encoded_start = self.tokenizer.convert_tokens_to_ids('[start]')
    self.encoded_end = self.tokenizer.convert_tokens_to_ids('[end]')
    self.target_num = target_num

  def __len__(self):
    return len(self.data)  

  def __getitem__(self, index):

    # pil_image
    # image_array
    # image_tensor
    pil_image = self.data[index][0]
    image_array = np.array(pil_image)
    copied_array = image_array.copy()
    image_tensor = torch.Tensor(copied_array)

    # normalization
    # normalized_tensor
    # feature_map
    image_size = (image_array.shape[0], image_array.shape[1])
    image_tensor = image_tensor.view(3, image_size[0], image_size[1])
    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)
    normalization = torchvision.transforms.Normalize(normalize_mean, normalize_std)
    normalized_tensor = normalization(image_tensor)
    normalized_tensor = normalized_tensor.unsqueeze(0)
    normalized_tensor = normalized_tensor.to(device)
    feature_map = self.feature_extractor(normalized_tensor)
    feature_map = feature_map.squeeze(0)

    # target_list
    # tokenized_caption
    # encoded_caption
    target_list = self.data[index][1]
    target_list = self.adjust_num(target_list, self.target_num)
    target_list = " ".join(target_list).lower()
    tokenized_caption = self.tokenizer.tokenize(target_list)
    encoded_caption = self.tokenizer.encode(tokenized_caption)

    # caption_ids
    # caption_target
    if len(encoded_caption) >= (self.max_len - 1):
      caption_ids = [self.encoded_start] + encoded_caption[:(self.max_len - 1)]
      caption_target = encoded_caption[:(self.max_len - 1)] + [self.encoded_end]

    else:
      caption_ids = [self.encoded_start] + encoded_caption
      caption_target = encoded_caption + [self.encoded_end]

    # padding_length
    # caption_ids
    # caption_target
    padding_length = self.max_len - len(caption_ids)
    caption_ids = self.padding(caption_ids, self.encoded_pad, padding_length)
    caption_ids = torch.Tensor(caption_ids)
    caption_ids = caption_ids.long()

    padding_length = self.max_len - len(caption_target)
    caption_target = self.padding(caption_target, self.encoded_pad, padding_length)
    caption_target = torch.Tensor(caption_target)
    caption_target = caption_target.long()

    # dictionary
    dictionary = {}
    dictionary['feature_map'] = feature_map
    dictionary['caption_ids'] = caption_ids
    dictionary['caption_target'] = caption_target

    return dictionary

  def adjust_num(self, input, num):
    output = []
    for i in range(num):
      output.append(input[i])
    return output

  def padding(self, input, value, length):
    return input + [value] * length
