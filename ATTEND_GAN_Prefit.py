#11) ATTEND_GAN_Prefit
 #11.1) pretrain generator
 #11.2) pretrain discriminator

def ATTEND_GAN_Genprefit(dataloader, model, loss_function, epoches, learning_rate):

  #11.1) pretrain generator
  optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
  for i in range(epoches):
    print(f"epoches:{i+1}")
    print('pretrain_generator')
    ATTEND_GAN_Pretrain.pretrain_generator(dataloader,
                                           model,
                                           loss_function,
                                           optimizer,
                                           1.0)
    torch.save(model, '/content/gdrive/My Drive/' + f'pretrained_generator:{i+1}')

def ATTEND_GAN_Disprefit(dataloader, model, loss_function, epochs, learning_rate):
  
  #11.2) pretrain discriminator
  optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
  for i in range(epochs):
    print(f"epoches:{i+1}")
    print('pretrain_discriminator')
    ATTEND_GAN_Pretrain.pretrain_discriminator(dataloader,
                                               model,
                                               loss_function,
                                               optimizer)
    torch.save(model, '/content/gdrive/My Drive/' + f'pretrained_discriminator:{i+1}')
