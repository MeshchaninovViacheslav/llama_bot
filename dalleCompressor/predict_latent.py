import torch
from tqdm import tqdm

from utils import maybe, tokenizer, resize_image_to, cast_tuple
from diffusion_model import step_dif

@torch.no_grad()
def predict_latent(
    target_image,
    decoder,
    unet,
    text_encodings,
    lowres_cond_img,
    noise_scheduler,
    timesteps
):
    total_timesteps, alphas= noise_scheduler.num_timesteps, noise_scheduler.alphas_cumprod
    times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1]
    times = list(reversed(times.int().tolist()))

    time_pairs = list(zip(times[:-1], times[1:]))
    time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))
    time_pairs = time_pairs[::-1]

    cond_img = maybe(decoder.normalize_img)(lowres_cond_img)
    target_image  = maybe(decoder.normalize_img)(target_image)
    img = target_image.detach().clone()

    trajectory = {
        "x_t" : [],
        "x_0" : [],
        "eps" : []
    }
    
    for time, time_next in tqdm(time_pairs):
        alpha = alphas[time]
        alpha_next = alphas[time_next]
        alpha, alpha_next = alpha_next, alpha

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

    latent = img
    
    return latent, trajectory

def generate_latent(target_image, decoder, cond_image, text):
    text = tokenizer.tokenize(text).cuda()

    start_at_unet_number = 2
    num_unets = decoder.num_unets

    _, text_encodings = decoder.clip.embed_text(text)
    
    prev_unet_output_size = decoder.image_sizes[start_at_unet_number - 2]
    cond_image = resize_image_to(cond_image, prev_unet_output_size, nearest = True)
    num_unets = decoder.num_unets
    
    for unet_number, unet, vae, image_size, noise_scheduler, sample_timesteps in \
        zip(range(1, num_unets + 1), decoder.unets, decoder.vaes, decoder.image_sizes, decoder.noise_schedulers, decoder.sample_timesteps):
        
        if unet_number < start_at_unet_number:
            continue  # It's the easiest way to do it

        # prepare low resolution conditioning for upsamplers
        image_size = vae.get_encoded_fmap_size(image_size)
        lowres_cond_img = resize_image_to(cond_image, target_image_size = image_size, clamp_range = decoder.input_image_range, nearest = True)

        latent, trajectory = predict_latent(
            target_image,
            decoder,
            unet,
            text_encodings = text_encodings,
            lowres_cond_img = lowres_cond_img,
            noise_scheduler = noise_scheduler,
            timesteps = sample_timesteps,
        )
    return latent, trajectory
