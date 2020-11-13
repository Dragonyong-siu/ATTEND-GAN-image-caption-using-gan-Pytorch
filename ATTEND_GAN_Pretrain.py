#10) ATTEND_GAN_Pretrain
 #10.1) pretrain caption generator using 7.1 : L2
 #10.2) pretrain caption discriminator using 7.3 

from tqdm import tqdm

class train_set():
  def pretrain_generator(self,
                         dataloader, 
                         generator, 
                         loss_function, 
                         optimizer, 
                         lambda1):

    #10.1) pretrain caption generator using 7.1 : L2
    generator.train()
    book = tqdm(dataloader, total = len(dataloader))
    total_gen_loss = 0.0
    for bi, dictionary in enumerate(book):
      feature_map = dictionary['feature_map']
      caption_ids = dictionary['caption_ids']
      caption_target = dictionary['caption_target']

      feature_map = feature_map.to(device)
      caption_ids = caption_ids.to(device)
      caption_target = caption_target.to(device)

      generator.zero_grad()
      logits, atts = generator(feature_map, caption_ids)

      logits = logits.view(-1, 50260)
      caption_target = caption_target.view(-1)

      generator_loss = loss_function(logits, atts, caption_target, lambda1)
      generator_loss.backward()

      optimizer.step()
      optimizer.zero_grad()
      total_gen_loss += generator_loss

    average_gen_loss = total_gen_loss / len(dataloader)
    print(" average_gen_loss: {0:.2f}".format(average_gen_loss))

  def pretrain_discriminator(self,
                             dataloader, 
                             discriminator,  
                             loss_function, 
                             optimizer):

    #10.2) pretrain caption discriminator using 7.3 
    discriminator.train()
    book = tqdm(dataloader, total = len(dataloader))
    total_dis_loss = 0.0
    for bi, dictionary in enumerate(book):
      input_data = dictionary['input_data']
      target_data = dictionary['target_data']

      input_data = input_data.to(device)
      target_data = target_data.to(device)

      discriminator.zero_grad()
      score = discriminator(input_data)
      
      discriminator_loss = loss_function(score, target_data)
      discriminator_loss.backward()

      optimizer.step()
      optimizer.zero_grad()
      total_dis_loss += discriminator_loss

    average_dis_loss = total_dis_loss / len(dataloader)
    print(" average_dis_loss: {0:.2f}".format(average_dis_loss))

ATTEND_GAN_Pretrain = train_set()
