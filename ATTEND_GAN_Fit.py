#13) ATTEND_GAN_Fit
 #13.1) train generator
 #13.2) train discriminator

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
