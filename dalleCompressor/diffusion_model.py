import torch
from tqdm import tqdm


from utils import maybe, cast_tuple, tokenizer, resize_image_to, set_seed

def forward_with_cond_scale(
        unet,
        *args,
        **kwargs
    ):
        logits = unet.forward(*args, **kwargs)
        return logits


def step_dif(
        img, 
        decoder, 
        unet, 
        alpha, 
        alpha_next, 
        time, 
        text_encodings, 
        cond_img
    ):
    time_cond = torch.full((img.shape[0],), time, dtype = torch.long).cuda()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16): 
        pred = forward_with_cond_scale(
            unet,
            img, 
            time_cond, 
            text_encodings=text_encodings, 
            lowres_cond_img=cond_img,
            image_embed=cond_img
        )

    sigma = 1 - alpha
    sigma_next = 1 - alpha_next
    x_0 = (img - sigma.sqrt() * pred) / alpha.sqrt()
    
    x_0 = decoder.dynamic_threshold(x_0)
    pred_noise = (img - x_0 * alpha.sqrt()) / sigma.sqrt()
    img = x_0 * alpha_next.sqrt() + pred_noise * sigma_next.sqrt()

    return {
        "x_t": img,
        "x_0": x_0,
        "eps": pred_noise
    }


@torch.no_grad()
def ddim(
    decoder,
    unet,
    text_encodings,
    lowres_cond_img,
    noise_scheduler,
    timesteps,
    latent=None,
):

    set_seed(0)

    total_timesteps, alphas = noise_scheduler.num_timesteps, noise_scheduler.alphas_cumprod
    times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1]

    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))

    cond_img = maybe(decoder.normalize_img)(lowres_cond_img)
    img = torch.randn_like(cond_img) if latent is None else latent
    
    for time, time_next in tqdm(time_pairs):
        alpha = alphas[time]
        alpha_next = alphas[time_next]
        
        img = step_dif(
            img=img, 
            decoder=decoder, 
            unet=unet, 
            alpha=alpha, 
            alpha_next=alpha_next, 
            time=time, 
            text_encodings=text_encodings,
            cond_img=cond_img,
        )["x_t"]
    
    img = decoder.unnormalize_img(img)
    return img


def generate(decoder, cond_image, text, latent):
    text = tokenizer.tokenize(text).cuda()
    start_at_unet_number = 2
    num_unets = decoder.num_unets
    _, text_encodings = decoder.clip.embed_text(text)
    
    for unet_number, unet, vae, image_size, noise_scheduler, sample_timesteps in \
        zip(range(1, num_unets + 1), decoder.unets, decoder.vaes, decoder.image_sizes, decoder.noise_schedulers, decoder.sample_timesteps):
        if unet_number < start_at_unet_number:
            continue  # It's the easiest way to do it

        # prepare low resolution conditioning for upsamplers
        image_size = vae.get_encoded_fmap_size(image_size)
        lowres_cond_img = resize_image_to(cond_image, target_image_size = image_size, clamp_range = decoder.input_image_range, nearest = True)

        img = ddim(
            decoder,
            unet,
            text_encodings = text_encodings,
            lowres_cond_img = lowres_cond_img,
            noise_scheduler = noise_scheduler,
            timesteps = sample_timesteps,
            latent=latent
        )
    
    return img