#8) ATTEND_GAN_Disdataset
 #8.1) prepare dataset for training discriminator : real_data 
 #8.2) prepare dataset for training discriminator : fake_data

class ATTEND_GAN_Disdataset_real(torch.utils.data.Dataset):
  def __init__(self, 
               data, 
               dataset, 
               sampler, 
               caption_length, 
               encoder, 
               tokenizer, 
               caption_num):
    
    self.data = data
    self.dataset = dataset
    self.sampler = sampler
    self.caption_length = caption_length
    self.encoder = encoder
    self.tokenizer = tokenizer
    self.caption_num = caption_num
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):

    #8.1) prepare dataset for training discriminator : real_data 
    # attend_gan_dataset
    attend_gan_dataset = self.dataset(self.data, 
                                      self.caption_length, 
                                      self.encoder,
                                      self.tokenizer,
                                      self.caption_num)
    
    # real_data
    real_data = attend_gan_dataset[index]
    real_data = real_data['caption_ids']

    # target_data for real_data
    target_data = torch.Tensor([1.0])
    target_data = target_data.to(device)

    # dictionary
    dictionary = {}
    dictionary['input_data'] = real_data
    dictionary['target_data'] = target_data

    return dictionary

class ATTEND_GAN_Disdataset_fake(torch.utils.data.Dataset):
  def __init__(self, 
               data, 
               dataset, 
               sampler, 
               caption_length, 
               encoder, 
               tokenizer, 
               caption_num):
    
    self.data = data
    self.dataset = dataset
    self.sampler = sampler
    self.caption_length = caption_length
    self.encoder = encoder
    self.tokenizer = tokenizer
    self.caption_num = caption_num
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):

    #8.2) prepare dataset for training discriminator : fake_data 
    # attend_gan_dataset
    attend_gan_dataset = self.dataset(self.data, 
                                      self.caption_length, 
                                      self.encoder,
                                      self.tokenizer,
                                      self.caption_num)
    
    # feature
    # fake_data
    feature = attend_gan_dataset[index]
    feature = feature['feature_map']
    fake_data = self.sampler(feature, None, self.caption_length)
    fake_data = fake_data[:1, :]
    fake_data = fake_data.squeeze(0)

    # target_data for fake_data
    target_data = torch.Tensor([0.0])
    target_data = target_data.to(device)

    # dictionary
    dictionary = {}
    dictionary['input_data'] = fake_data
    dictionary['target_data'] = target_data

    return dictionary
