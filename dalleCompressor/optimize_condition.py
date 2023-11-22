import torch
from tqdm import tqdm

from diffusion_model import tokenizer, maybe, step_dif
from utils import set_seed, resize_image_to


def optimize(decoder, text, init_image, target_image, image):
    text = tokenizer.tokenize(text).cuda()
    cond_scale = 1.

    batch_size = init_image.shape[0]
    start_at_unet_number = 2
    num_unets = decoder.num_unets
    _, text_encodings = decoder.clip.embed_text(text)
    
    for unet_number, unet, vae, channel, image_size, noise_scheduler, sample_timesteps in \
        zip(range(1, num_unets + 1), decoder.unets, decoder.vaes, decoder.sample_channels, decoder.image_sizes, decoder.noise_schedulers, decoder.sample_timesteps):
        if unet_number < start_at_unet_number:
            continue  # It's the easiest way to do it
            
        learn_image = torch.tensor(image.data)
        learn_image.requires_grad = True
        
        optimizer = torch.optim.Adam(
            params=[learn_image],
            lr=0.1,
            betas=(0.8, 0.8)
        )
        
        def prep_image(image):
            up_img = resize_image_to(image, target_image_size = image_size, clamp_range = decoder.input_image_range, nearest = True)
            up_img = maybe(vae.encode)(up_img)
            return up_img

        # prepare low resolution conditioning for upsamplers
        
        shape = (batch_size, channel, image_size, image_size)
        image_size = vae.get_encoded_fmap_size(image_size)
        shape = (batch_size, vae.encoded_dim, image_size, image_size)
        
        set_seed(0)

        timesteps = sample_timesteps
        batch, device, total_timesteps, alphas, eta = shape[0], decoder.device, noise_scheduler.num_timesteps, noise_scheduler.alphas_cumprod, decoder.ddim_sampling_eta
        times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))
        time_pairs = time_pairs[::-1]

        x_t = torch.randn(shape, device = device) if init_image is None else init_image
        
        num_epochs = 10
        T = tqdm(range(num_epochs))
        
        for _ in T:
            for time, time_next in time_pairs:
                alpha = alphas[time]
                alpha_next = alphas[time_next]

                def f_loss():
                    optimizer.zero_grad()
                    lowres_cond_img = prep_image(learn_image)
                    pred_dict = step_dif(
                        x_t, decoder, unet, alpha, alpha_next, time, batch, device, 
                        lowres_cond_img, text_encodings, cond_scale, lowres_cond_img, None, noise_scheduler
                    )
                    
                    loss = torch.mean((pred_dict["x_t"] - target_image) ** 2) 
                    loss.backward()
                    T.set_description(f"loss: {loss.item():0.5f}")           
                    return loss

                for _ in range(10):
                    optimizer.step(f_loss)

                lowres_cond_img = prep_image(learn_image)
                x_t = step_dif(
                    x_t, decoder, unet, alpha, alpha_next, time, batch, device, 
                    lowres_cond_img, text_encodings, cond_scale, lowres_cond_img, None, noise_scheduler
                )["x_t"].detach()

        x_t = vae.decode(x_t)
    return x_t, learn_image