import torch
from tqdm import tqdm

from utils import maybe, tokenizer, resize_image_to, cast_tuple

@torch.no_grad()
def predict_latent(
    target_image,
    decoder,
    unet,
    shape,
    image_embed,
    text_encodings,
    cond_scale,
    lowres_cond_img,
    lowres_noise_level,
    noise_scheduler,
    timesteps
):
    batch, device, total_timesteps, alphas, eta = shape[0], decoder.device, noise_scheduler.num_timesteps, noise_scheduler.alphas_cumprod, decoder.ddim_sampling_eta
    times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1]
    times = list(reversed(times.int().tolist()))

    time_pairs = list(zip(times[:-1], times[1:]))
    time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))
    time_pairs = time_pairs[::-1]

    img = target_image.detach().clone() 
    lowres_cond_img = maybe(decoder.normalize_img)(lowres_cond_img)

    trajectory = {
        "x_t" : [],
        "x_0" : [],
        "eps" : []
    }
    
    for time, time_next in tqdm(time_pairs):
        alpha = alphas[time]
        alpha_next = alphas[time_next]
        alpha, alpha_next = alpha_next, alpha

        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16): 
            pred = unet.forward_with_cond_scale(
                img, 
                time_cond, 
                image_embed = image_embed, 
                text_encodings = text_encodings, 
                cond_scale = cond_scale, 
                lowres_cond_img = lowres_cond_img, 
                lowres_noise_level = lowres_noise_level
            )

        
        sigma = (1 - alpha).sqrt()
        sigma_next = (1 - alpha_next).sqrt()
        x_0 = (img - sigma * pred) / alpha.sqrt()
        
        x_0 = decoder.dynamic_threshold(x_0)
        pred_noise = (img - x_0 * alpha.sqrt()) / sigma
        img = x_0 * alpha_next.sqrt() + pred_noise * sigma_next
        
        trajectory["x_0"].append(x_0)
        trajectory["eps"].append(pred_noise)
        trajectory["x_t"].append(img)

    latent = img
    
    return latent, trajectory

def generate_latent(target_image, decoder, image, image_embed, text):
    text = tokenizer.tokenize(text).cuda()
    cond_scale = 1

    batch_size = image_embed.shape[0]
    start_at_unet_number = 2
    num_unets = decoder.num_unets

    _, text_encodings = decoder.clip.embed_text(text)
    
    prev_unet_output_size = decoder.image_sizes[start_at_unet_number - 2]
    img = resize_image_to(image, prev_unet_output_size, nearest = True)
    
    cond_scale = cast_tuple(cond_scale, num_unets)
    num_unets = decoder.num_unets
    
    for unet_number, unet, vae, channel, image_size, noise_scheduler, sample_timesteps, unet_cond_scale in \
        zip(range(1, num_unets + 1), decoder.unets, decoder.vaes, decoder.sample_channels, decoder.image_sizes, decoder.noise_schedulers, decoder.sample_timesteps, cond_scale):
        
        if unet_number < start_at_unet_number:
            continue  # It's the easiest way to do it

        # prepare low resolution conditioning for upsamplers
        lowres_cond_img = lowres_noise_level = None
        shape = (batch_size, channel, image_size, image_size)
        lowres_cond_img = resize_image_to(img, target_image_size = image_size, clamp_range = decoder.input_image_range, nearest = True)
        image_size = vae.get_encoded_fmap_size(image_size)
        shape = (batch_size, vae.encoded_dim, image_size, image_size)
        lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

        latent, trajectory = predict_latent(
            target_image,
            decoder,
            unet,
            shape,
            image_embed = image_embed,
            text_encodings = text_encodings,
            cond_scale = unet_cond_scale,
            lowres_cond_img = lowres_cond_img,
            lowres_noise_level = lowres_noise_level,
            noise_scheduler = noise_scheduler,
            timesteps = sample_timesteps,
        )
    return latent, trajectory
