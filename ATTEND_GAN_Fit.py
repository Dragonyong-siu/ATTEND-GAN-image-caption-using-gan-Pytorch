#13) ATTEND_GAN_Fit
 #13.1) train generator
 #13.2) train discriminator
 #13.3) ATTEND_GAN_Adverfit - algorithm 1
  #13.3.1) train generator g_steps
  #13.3.2) call ATTEND_GAN_Generator
  #13.3.3) rename disdataloader
  #13.3.4) train discriminator d_steps
  #13.3.5) call ATTEND_GAN_Discriminator
  #13.3.6) repeat those again

def ATTEND_GAN_Genfit(dataloader, 
                      generator,
                      discriminator, 
                      loss_function1,
                      loss_function2,
                      reward_function,
                      lambda1,
                      lambda2,
                      caption_length,
                      mc_num,
                      epoches,
                      learning_rate):
  
  #13.1) train generator
  optimizer = torch.optim.AdamW(generator.parameters(), lr = learning_rate)
  for i in range(epoches):
    print(f"epoches:{i+1}")
    print('train_generator')
    ATTEND_GAN_Advertrain.advertrain_generator(dataloader,
                                               generator,
                                               discriminator,
                                               loss_function1,
                                               loss_function2,
                                               reward_function,
                                               optimizer,
                                               lambda1,
                                               lambda2,
                                               caption_length,
                                               mc_num)
    torch.save(generator, '/content/gdrive/My Drive/' + f'trained_generator:{i+1}')

def ATTEND_GAN_Disfit(dataloader,
                      discriminator,
                      loss_function,
                      epoches,
                      learning_rate):
  
  #13.2) train discriminator
  optimizer = torch.optim.AdamW(discriminator.parameters(), lr = learning_rate)
  for i in range(epoches):
    print(f"epoches:{i+1}")
    print('train_discriminator')
    ATTEND_GAN_Advertrain.advertrain_discriminator(dataloader,
                                                   discriminator,
                                                   loss_function,
                                                   optimizer)
    torch.save(discriminator, '/content/gdrive/My Drive/' + f'trained_discriminator:{i+1}')

def ATTEND_GAN_Adverfit(repeat_num,
                        dataloader,
                        generator,
                        discriminator,
                        loss_function1,
                        loss_function2,
                        loss_function3,
                        reward_function,
                        lambda1,
                        lambda2,
                        caption_length,
                        mc_num,
                        g_steps,
                        d_steps,
                        learning_rate1,
                        learning_rate2,
                        batch_size):
  
  #13.3) ATTEND_GAN_Adverfit - algorithm 1
  #13.3.6) repeat those again
  for i in range(repeat_num):
    
    #13.3.1) train generator g_steps
    ATTEND_GAN_Genfit(dataloader, 
                      generator,
                      discriminator, 
                      loss_function1,
                      loss_function2,
                      reward_function,
                      lambda1,
                      lambda2,
                      caption_length,
                      mc_num,
                      g_steps,
                      learning_rate1)
    
    #13.3.2) call ATTEND_GAN_Generator
    ATTEND_GAN_Generator = torch.load('/content/gdrive/My Drive/' + f'trained_generator:{g_steps}')

    #13.3.3) rename disdataloader
    ATTEND_GAN_Disdataset = ATTEND_GAN_Disdataset_real(train_data, 
                                                       ATTEND_GAN_Dataset,
                                                       ATTEND_GAN_Generator.caption_sampler,
                                                       caption_length,
                                                       ATTEND_GAN_Encoder,
                                                       GPT2_Tokenizer,
                                                       batch_size) + \
                            ATTEND_GAN_Disdataset_fake(train_data, 
                                                       ATTEND_GAN_Dataset,
                                                       ATTEND_GAN_Generator.caption_sampler,
                                                       caption_length,
                                                       ATTEND_GAN_Encoder,
                                                       GPT2_Tokenizer,
                                                       batch_size)
                        
    train_disdataloader = DataLoader(ATTEND_GAN_Disdataset,
                                     batch_size = batch_size,
                                     shuffle = True,
                                     drop_last = True)
    
    #13.3.4) train discriminator d_steps
    ATTEND_GAN_Disfit(train_disdataloader, 
                      discriminator, 
                      loss_function3, 
                      d_steps, 
                      learning_rate2)
    
    #13.3.5) call ATTEND_GAN_Discriminator
    ATTEND_GAN_Discriminator = torch.load('/content/gdrive/My Drive/' + f'trained_discriminator:{d_steps}')
