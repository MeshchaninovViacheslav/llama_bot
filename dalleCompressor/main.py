import torch
import argparse

from dalle_init import decoder_init
from utils import read_image, PSNR
from predict_latent import generate_latent
from diffusion_model import generate
from optimize_condition import optimize


class LatentModel:
    def __init__(self, decoder_path):
        self.decoder = decoder_init(decoder_path)


    def fit(self, image_path, text_prompt=""):
        image_256 = [read_image(path=image_path, size=256)]
        image_64 = [read_image(path=image_path, size=64)]

        cond_image = torch.tensor(image_64, dtype=torch.float32).permute(0, 3, 1, 2).cuda()
        target_image = torch.tensor(image_256, dtype=torch.float32).permute(0, 3, 1, 2).cuda()

        text_prompt = [text_prompt]

        latent, _ = generate_latent(target_image, self.decoder, cond_image, cond_image, text_prompt)

        return {
            "text_prompt": text_prompt,
            "latent": latent,
            "cond_image": cond_image,
        }
    

    def predict(self, latent, cond_image, text_prompt, psnr_need=True, target_image=None):
        pred_image = generate(self.decoder, cond_image, cond_image, text_prompt, latent)
        pred_image = pred_image.permute(0, 2, 3, 1).cpu().numpy()[0]

        result = {
            "pred_image": pred_image
        }
        if psnr_need:
            result["PSNR"] = PSNR(pred_image, target_image)

        return result
    

    def fit_predict(self, image_path, text_prompt="", psnr_need=True):
        image_256 = [read_image(path=image_path, size=256)]
        image_64 = [read_image(path=image_path, size=64)]

        cond_image = torch.tensor(image_64, dtype=torch.float32).permute(0, 3, 1, 2).cuda()
        target_image = torch.tensor(image_256, dtype=torch.float32).permute(0, 3, 1, 2).cuda()

        text_prompt = [text_prompt]

        latent, _ = generate_latent(target_image, self.decoder, cond_image, cond_image, text_prompt)

        latent = torch.nn.functional.interpolate(latent, size=(64, 64), mode="area")
        init_image = torch.nn.functional.interpolate(latent, size=(256, 256), mode="area")

        _, learn_image = optimize(self.decoder, text_prompt, init_image, target_image, cond_image)
        pred_image = generate(self.decoder, learn_image, learn_image, text_prompt, init_image)
        pred_image = pred_image.permute(0, 2, 3, 1).cpu().numpy()

        result = {
            "pred_image": pred_image[0],
            "latent": latent,
        }
        if psnr_need:
            result["PSNR"] = PSNR(pred_image[0], image_256[0])
            print("PSNR = ", result["PSNR"])

        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--decoder_path', default="weights/second_decoder")
    parser.add_argument('-i', '--image_path', default="test_images/1.jpeg")
    parser.add_argument('-t', '--text_prompt', default="")

    args = parser.parse_args()
    print(args)
    
    model = LatentModel(args.decoder_path)
    result = model.fit_predict(args.image_path, args.text_prompt)



main()