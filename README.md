# ATTEND_GAN-image_caption_using_gan_Pytorch

Towards_Generating_Stylized_Image_Captions_via_Adversarial_Training
https://arxiv.org/pdf/1908.02943.pdf



 0) download coco_dataset.zip and unzip

 1) data_transformation

     1.1) resize to (224, 224, 3)
 
     1.2) make couple :(image, caption_target)

 2) ATTEND_GAN_Encoder
  
      using pretrained ResNet-152(ImageNet dataset)
  
      usinf Res5c layer to extract the spatial features
  
      7 * 7 * 2048 feature_map 

 3) ATTEND_GAN_Dataset
 
     3.1) ATTEND_GAN_Dataset_LSTM
  
     feature_map from pretrained ResNet-152
  
     using gpt2_tokenizer & gpt2_encoder
  
     feature_map
  
     caption_ids
  
     caption_target

 4) ATTEND_GAN_DataLoader
 
     4.1) loader_dataset
 
     4.2) train_dataloader


 5) ATTEND_GAN_Generator
 
     5.1) using lstm for pretrain & train generator
 
     5.2) caption_sampler
 
     5.3) generator_loss(lambda * L_1 + L_2)
 
 6) Caption_Discriminator
 
     6.1) using gru to get hidden_state and then get score between [0, 1]
 
     6.2) discriminator_loss

 7) ATTEND_GAN_Loss
 
     7.1) generator_loss : L2
 
        7.1.1) crossentropy function 
 
        7.1.2) (1 - x)^2 function


     7.2) generator_loss : L1
 
        7.2.1) connect with discriminator : reward


     7.3) discriminator_loss : binary crossentropy function

     7.4) get_reward : calculate policy gradient to get reward

        7.4.1) iterate MC_num times to decrease the variance of the next words


 8) ATTEND_GAN_Disdataset
 
     8.1) prepare dataset for training discriminator : real_data 
 
     8.2) prepare dataset for training discriminator : fake_data

 9) ATTEND_GAN_DisdataLoader
 
     9.1) ATTEND_GAN_Disdataset
 
     9.2) train_disdataloader

 10) ATTEND_GAN_Pretrain
 
     10.1) pretrain caption generator using 7.1 : L2
 
     10.2) pretrain caption discriminator using 7.3 

 11) ATTEND_GAN_Prefit
 
     11.1) pretrain generator
 
     11.2) pretrain discriminator
     
 12) ATTEND_GAN_Advertrain
     
     12.1) adversarially train generator using 7.1 & 7.2
     
         12.1.1) get generator_loss2 using L2
     
         12.1.2) get samples from generator & get rewards using reward_function
        
         12.1.3) get generator_loss2
 
     12.2) get total_loss by sum generator_loss1 and generator_loss1

     12.3) make new dataset & dataloader from trained generator g-steps

     12.4) adversarially train discriminator using 7.3 
     
     12.5) repeat (g steps for generator, d steps for discriminator)

13) ATTEND_GAN_Fit

     13.1) train generator
     
     13.2) train discriminator
     
     13.3) ATTEND_GAN_Adverfit - algorithm 1
     
        13.3.1) train generator g_steps
        
        13.3.2) call ATTEND_GAN_Generator
        
        13.3.3) rename disdataloader
        
        13.3.4) train discriminator d_steps
        
        13.3.5) call ATTEND_GAN_Discriminator
        
        13.3.6) repeat those again
