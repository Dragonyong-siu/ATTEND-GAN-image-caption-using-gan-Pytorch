#9) ATTEND_GAN_DisdataLoader
 #9.1) ATTEND_GAN_Disdataset
 #9.2) train_disdataloader

from torch.utils.data import DataLoader
ATTEND_GAN_Disdataset = ATTEND_GAN_Disdataset_real(train_data, 
                                                   ATTEND_GAN_Dataset,
                                                   ATTEND_GAN_Generator.caption_sampler,
                                                   16,
                                                   ATTEND_GAN_Encoder,
                                                   GPT2_Tokenizer,
                                                   2) + \
                        ATTEND_GAN_Disdataset_fake(train_data, 
                                                   ATTEND_GAN_Dataset,
                                                   ATTEND_GAN_Generator.caption_sampler,
                                                   16,
                                                   ATTEND_GAN_Encoder,
                                                   GPT2_Tokenizer,
                                                   2)
                        
train_disdataloader = DataLoader(ATTEND_GAN_Disdataset,
                                 batch_size = 2,
                                 shuffle = True,
                                 drop_last = True)
