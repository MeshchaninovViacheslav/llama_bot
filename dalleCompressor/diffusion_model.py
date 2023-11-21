import torch
from tqdm import tqdm


from utils import maybe, cast_tuple, tokenizer, resize_image_to, set_seed

def forward_with_cond_scale(
        unet,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = unet.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits

        null_logits = unet.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

def step_dif(img, decoder, unet, alpha, alpha_next, time, batch, device, image_embed, text_encodings, cond_scale, lowres_cond_img, lowres_noise_level, noise_scheduler):
    time_cond = torch.full((batch,), time, device=device, dtype = torch.long)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16): 
        pred = forward_with_cond_scale(
            unet,
            img, 
            time_cond, 
            image_embed = image_embed, 
            text_encodings = text_encodings, 
            cond_scale = cond_scale, 
            lowres_cond_img = lowres_cond_img, 
            lowres_noise_level = lowres_noise_level
        )
    
    x_0 = noise_scheduler.predict_start_from_noise(img, t=time_cond, noise=pred)
    x_0 = decoder.dynamic_threshold(x_0)

    sigma = (1 - alpha).sqrt()
    sigma_next = (1 - alpha_next).sqrt()
    x_0 = (img - sigma * pred) / alpha.sqrt()
    x_0 = decoder.dynamic_threshold(x_0)
    pred_noise = (img - x_0 * alpha.sqrt()) / sigma
    img = x_0 * alpha_next.sqrt() + pred_noise * sigma_next
    return {
        "x_t": img,
        "x_0": x_0,
        "eps": pred_noise
    }

@torch.no_grad()
def ddim(
    decoder,
    unet,
    shape,
    image_embed,
    text_encodings,
    cond_scale,
    lowres_cond_img,
    lowres_noise_level,
    noise_scheduler,
    timesteps,
    latent=None,
):

    set_seed(0)

    batch, device, total_timesteps, alphas, eta = shape[0], decoder.device, noise_scheduler.num_timesteps, noise_scheduler.alphas_cumprod, decoder.ddim_sampling_eta
    times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1]

    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))

    img = torch.randn(shape, device = device) if latent is None else latent
    cond_img = maybe(decoder.normalize_img)(lowres_cond_img)
    
    for time, time_next in tqdm(time_pairs):
        alpha = alphas[time]
        alpha_next = alphas[time_next]

        img = step_dif(
            img, decoder, unet, alpha, alpha_next, time, batch, device, 
            image_embed, text_encodings, cond_scale, cond_img, lowres_noise_level, noise_scheduler
        )["x_t"]
    return img


def generate(decoder, image, image_embed, text, latent):
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
    
    for unet_number, unet, vae, channel, image_size, noise_scheduler, lowres_cond, sample_timesteps, unet_cond_scale in \
        zip(range(1, num_unets + 1), decoder.unets, decoder.vaes, decoder.sample_channels, decoder.image_sizes, decoder.noise_schedulers, decoder.lowres_conds, decoder.sample_timesteps, cond_scale):
        if unet_number < start_at_unet_number:
            continue  # It's the easiest way to do it

        # prepare low resolution conditioning for upsamplers
        lowres_cond_img = lowres_noise_level = None
        shape = (batch_size, channel, image_size, image_size)
        lowres_cond_img = resize_image_to(img, target_image_size = image_size, clamp_range = decoder.input_image_range, nearest = True)
        image_size = vae.get_encoded_fmap_size(image_size)
        shape = (batch_size, vae.encoded_dim, image_size, image_size)
        lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

        img = ddim(
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
            latent=latent
        )

        img = vae.decode(img)
    return img