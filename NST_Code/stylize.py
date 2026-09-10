"""Command-line AdaIN style transfer.

Runs the same inference path as the web app, without Flask:

    content ---> VGG encoder (relu4_1) ---+
                                          +--> AdaIN --> alpha blend --> decoder --> output
    style   ---> VGG encoder (relu4_1) ---+

Examples
--------
    python stylize.py --content examples/brad_pitt.jpg --style examples/sketch.png
    python stylize.py --content examples/brad_pitt.jpg --style style_data/mondrian.jpg \
                      --alpha 0.0 0.25 0.5 0.75 1.0
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, style_interpolation

BASE_DIR = Path(__file__).resolve().parent


def parse_arguments():
    parser = argparse.ArgumentParser(description='AdaIN style transfer (command line)')

    parser.add_argument('--content', type=str, required=True,
                        help='Path to the content image')
    parser.add_argument('--style', type=str, nargs='+', required=True,
                        help='Path to the style image. Several paths interpolate between styles.')
    parser.add_argument('--style_weights', type=float, nargs='+', default=None,
                        help='Interpolation weight per style image. Defaults to equal weights.')
    parser.add_argument('--alpha', type=float, nargs='+', default=[1.0],
                        help='Style strength in [0, 1]. Accepts several values.')

    parser.add_argument('--output_dir', type=str, default=str(BASE_DIR / 'output'),
                        help='Directory to write results into')
    parser.add_argument('--size', type=int, default=512,
                        help='Resize the shorter side of both images to this before encoding')

    parser.add_argument('--vgg', type=str, default=str(BASE_DIR / 'vgg_normalised.pth'),
                        help='Location of pre-trained VGG')
    parser.add_argument('--decoder', type=str,
                        default=str(BASE_DIR / 'experiment' / 'final_exp' / 'decoder_final.pth'),
                        help='Location of trained decoder')

    return parser.parse_args()


def style_transfer(content_image, style_images, style_weights, encoder, decoder, alpha, device, size):
    content_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_images = [style_transform(s).unsqueeze(0).to(device) for s in style_images]

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = [encoder(s, is_test=True) for s in style_images]

        if len(style_feats) == 1:
            stylized_feats = adaptive_instance_normalization(content_feats, style_feats[0])
        else:
            stylized_feats = style_interpolation(content_feats, style_feats, style_weights)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(args.decoder, map_location=device))

    encoder.eval()
    decoder.eval()

    content_path = Path(args.content)
    style_paths = [Path(s) for s in args.style]
    content_image = Image.open(content_path).convert('RGB')
    style_images = [Image.open(s).convert('RGB') for s in style_paths]

    style_weights = args.style_weights
    if style_weights is None:
        style_weights = [1.0 / len(style_paths)] * len(style_paths)
    if len(style_weights) != len(style_paths):
        raise ValueError('Give one --style_weights value per --style image')
    total = sum(style_weights)
    style_weights = [w / total for w in style_weights]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for alpha in args.alpha:
        stylized_image = style_transfer(content_image, style_images, style_weights,
                                        encoder, decoder, alpha, device, args.size)

        styles = '_'.join(f'{p.stem}{w:g}' for p, w in zip(style_paths, style_weights))             if len(style_paths) > 1 else style_paths[0].stem
        name = f'{content_path.stem}_stylized_by_{styles}_alpha{alpha:g}.jpg'
        output_path = output_dir / name
        save_image(stylized_image, output_path)
        print(f'alpha={alpha:<4g} -> {output_path}')


if __name__ == '__main__':
    main()
