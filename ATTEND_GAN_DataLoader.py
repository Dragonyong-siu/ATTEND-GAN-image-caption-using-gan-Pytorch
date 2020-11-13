#4) ATTEND_GAN_DataLoader
 #4.1) loader_dataset
 #4.2) train_dataloader

from torch.utils.data import DataLoader
loader_dataset = ATTEND_GAN_Dataset(data = train_data,
                                    max_len = 16,
                                    feature_extractor = ATTEND_GAN_Encoder,
                                    tokenizer = GPT2_Tokenizer,
                                    target_num = 2)
train_dataloader = DataLoader(loader_dataset,
                              batch_size = 2,
                              shuffle = True,
                              drop_last = True)
