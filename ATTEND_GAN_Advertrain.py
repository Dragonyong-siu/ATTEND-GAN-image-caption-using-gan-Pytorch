#12) ATTEND_GAN_Advertrain
 #12.1) adversarially train generator using 7.1 & 7.2
  #12.1.1) get generator_loss2 using L2
  #12.1.2) get samples from generator & get rewards using reward_function
  #12.1.3) get generator_loss2
 
 #12.2) get total_loss by sum generator_loss1 and generator_loss1

 #12.3) make new dataset & dataloader from trained generator g-steps

 #12.4) adversarially train discriminator using 7.3 
 #12.5) repeat (g steps for generator, d steps for discriminator)

class adversarial_trainset():
  def advertrain_generator(self,
                           dataloader,
                           generator,
                           discriminator,
                           loss_function1,
                           loss_function2,
                           reward_function,
                           optimizer, 
                           lambda1,
                           lambda2,
                           caption_length,
                           mc_num):
    
    #12.1) adversarially train generator using 7.1 & 7.2
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

      #12.1) adversarially train generator using 7.1 & 7.2
       #12.1.1) get generator_loss2 using L2
      generator_loss2 = loss_function2(logits, atts, caption_target, lambda1)

      #12.1) adversarially train generator using 7.1 & 7.2
       #12.1.2) get samples from generator & get rewards using reward_function
      samples = generator.caption_sampler(feature_map, None, caption_length)
      rewards = reward_function(samples, mc_num, feature_map, generator, discriminator)

      #12.1) adversarially train generator using 7.1 & 7.2
       #12.1.3) get generator_loss1
      samples = samples.view(-1)
      samples = samples.to(device)

      rewards = rewards.view(-1)
      rewards = rewards.to(device)
      
      generator_loss1 = loss_function1(logits, samples, rewards)

      #12.2) get total_loss by sum generator_loss1 and generator_loss2
      loss_sum = torch.mul(lambda2, generator_loss1) + generator_loss2
      loss_sum.backward()

      optimizer.step()
      optimizer.zero_grad()
      total_gen_loss += loss_sum
    
    average_gen_loss = total_gen_loss / len(dataloader)
    print(" average_gen_loss: {0:.2f}".format(average_gen_loss))
  
  def advertrain_discriminator(self,
                               dataloader,
                               discriminator,
                               loss_function,
                               optimizer):
    
    #12.4) adversarially train discriminator using 7.3 
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

ATTEND_GAN_Advertrain = adversarial_trainset()  
