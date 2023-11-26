import torch
from tqdm import tqdm

from diffusion_model import tokenizer, maybe, step_dif
from utils import set_seed, resize_image_to, PSNR


def optimize(decoder, text, latent, target_image, cond_image):
    text = tokenizer.tokenize(text).cuda()
    start_at_unet_number = 2
    num_unets = decoder.num_unets
    _, text_encodings = decoder.clip.embed_text(text)
    
    for unet_number, unet, vae, image_size, noise_scheduler, sample_timesteps in \
        zip(range(1, num_unets + 1), decoder.unets, decoder.vaes, decoder.image_sizes, decoder.noise_schedulers, decoder.sample_timesteps):
        if unet_number < start_at_unet_number:
            continue  # It's the easiest way to do it
            
        learn_image = torch.tensor(cond_image.data)
        learn_image.requires_grad = True
        # learn_latent = torch.tensor(latent.data)
        # learn_latent.requires_grad = True

        optimizer = torch.optim.SGD(
            params=[learn_image],
            lr=2,
        )
        
        def prep_image(image):
            up_img = resize_image_to(image, target_image_size = image_size, clamp_range = decoder.input_image_range, nearest = True)
            up_img = maybe(decoder.normalize_img)(up_img)
            return up_img

        # prepare low resolution conditioning for upsamplers
        image_size = vae.get_encoded_fmap_size(image_size)
        target_image_normalized = prep_image(target_image)
        
        timesteps = sample_timesteps
        total_timesteps, alphas = noise_scheduler.num_timesteps, noise_scheduler.alphas_cumprod
        times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))
        
        num_epochs = 100
        T = tqdm(range(num_epochs))

        best_psnr = 0.
        best_image = None

        for epoch in T:
            set_seed(0)
            x_t = torch.randn_like(target_image)# if latent is None else latent.clone()
            
            loss = None
            loss_steps = []

            for time, time_next in time_pairs:
                optimizer.zero_grad()
                alpha = alphas[time]
                alpha_next = alphas[time_next]

                lowres_cond_img = prep_image(learn_image)

                pred_dict = step_dif(
                    img=x_t, 
                    decoder=decoder, 
                    unet=unet, 
                    alpha=alpha, 
                    alpha_next=alpha_next, 
                    time=time,  
                    text_encodings=text_encodings, 
                    cond_img=lowres_cond_img
                )

                # loss_cur = torch.mean((pred_dict["x_0"] - target_image_normalized) ** 2)
                # if loss is None:
                #     loss = loss_cur
                # else:
                #     loss += loss_cur
                loss = torch.mean((pred_dict["x_0"] - target_image_normalized) ** 2)

                loss.backward()
                optimizer.step()
                
                x_t = pred_dict["x_t"].detach()
                #loss_steps.append(loss_cur.item())
                
                

            #print(torch.mean(learn_image.grad ** 2).sqrt())

            img = decoder.unnormalize_img(x_t)[0].permute(1, 2, 0).cpu().numpy()
            t_img = target_image[0].permute(1, 2, 0).cpu().numpy()
            psnr = PSNR(img, t_img)
            if psnr > best_psnr:
                best_psnr = psnr
                best_image = learn_image.clone()
            #print("img", img.mean())
            T.set_description(f"Epoch: {epoch} loss: {loss.item():0.3f}, psnr: {psnr:0.3f}")           
            #print(f"Epoch: {epoch} loss: {loss.item():0.5f}, psnr: {PSNR(img, t_img):0.3f}")
            #print(" ".join([f"{l:0.2f}" for l in loss_steps]))
    # print(torch.min(learn_image), torch.max(learn_image))

    print(f"Best PSNR: {best_psnr: 0.3f}")
    return best_image