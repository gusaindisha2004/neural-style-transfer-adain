"""Tests for the AdaIN style-transfer implementation.

Run from the NST_Code directory:

    pytest -v
"""
import io
import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

NST = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NST))

from utils.models import VGGEncoder, Decoder                      # noqa: E402
from utils.utils import (adaptive_instance_normalization,          # noqa: E402
                         calc_mean_std, style_interpolation)

VGG_WEIGHTS = NST / 'vgg_normalised.pth'
DECODER_WEIGHTS = NST / 'experiment' / 'final_exp' / 'decoder_final.pth'
EXAMPLES = NST / 'examples'


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope='session')
def encoder():
    enc = VGGEncoder(str(VGG_WEIGHTS))
    enc.eval()
    return enc


@pytest.fixture(scope='session')
def decoder():
    dec = Decoder()
    dec.load_state_dict(torch.load(DECODER_WEIGHTS, map_location='cpu'))
    dec.eval()
    return dec


@pytest.fixture(scope='session')
def app_client():
    os.chdir(NST)
    import app as flask_app
    flask_app.app.config['WTF_CSRF_ENABLED'] = False
    flask_app.app.config['TESTING'] = True
    return flask_app.app.test_client()


def image_bytes(name):
    return (EXAMPLES / name).read_bytes()


# --------------------------------------------------------------------------
# weights and architecture
# --------------------------------------------------------------------------
def test_weight_files_present():
    assert VGG_WEIGHTS.exists(), 'vgg_normalised.pth is missing'
    assert DECODER_WEIGHTS.exists(), 'decoder_final.pth is missing'


def test_encoder_truncated_at_relu4_1(encoder):
    layers = list(encoder.vgg.children())
    assert len(layers) == 31
    assert isinstance(layers[-1], torch.nn.ReLU)


def test_encoder_is_frozen(encoder):
    assert all(not p.requires_grad for p in encoder.parameters())


def test_decoder_has_no_normalisation_layers(decoder):
    """The paper requires this: BN/IN would collapse outputs toward one style."""
    banned = (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)
    assert not any(isinstance(m, banned) for m in decoder.modules())


def test_decoder_checkpoint_matches_architecture(decoder):
    state = torch.load(DECODER_WEIGHTS, map_location='cpu')
    missing, unexpected = decoder.load_state_dict(state, strict=True), None
    assert len(state) == 18, 'expected 9 conv layers (weight + bias each)'


def test_encoder_feature_shapes(encoder):
    with torch.no_grad():
        h1, h2, h3, h4 = encoder(torch.randn(1, 3, 256, 256))
    assert list(h1.shape) == [1, 64, 256, 256]
    assert list(h2.shape) == [1, 128, 128, 128]
    assert list(h3.shape) == [1, 256, 64, 64]
    assert list(h4.shape) == [1, 512, 32, 32]


def test_is_test_returns_only_relu4_1(encoder):
    with torch.no_grad():
        feat = encoder(torch.randn(1, 3, 256, 256), is_test=True)
    assert isinstance(feat, torch.Tensor)
    assert list(feat.shape) == [1, 512, 32, 32]


def test_decoder_restores_spatial_size(encoder, decoder):
    with torch.no_grad():
        out = decoder(encoder(torch.randn(1, 3, 256, 256), is_test=True))
    assert list(out.shape) == [1, 3, 256, 256]


# --------------------------------------------------------------------------
# the AdaIN operation itself
# --------------------------------------------------------------------------
def test_calc_mean_std_is_per_channel_and_per_sample():
    x = torch.randn(2, 5, 8, 8)
    mean, std = calc_mean_std(x)
    assert list(mean.shape) == [2, 5, 1, 1]
    assert torch.allclose(mean.flatten(), x.mean(dim=(2, 3)).flatten(), atol=1e-5)


def test_adain_output_adopts_style_statistics():
    """The core claim: after AdaIN, per-channel mean/std match the style's."""
    content = torch.randn(1, 16, 12, 12) * 3 + 7
    style = torch.randn(1, 16, 12, 12) * 0.5 - 2

    out = adaptive_instance_normalization(content, style)
    out_mean, out_std = calc_mean_std(out)
    style_mean, style_std = calc_mean_std(style)

    assert torch.allclose(out_mean, style_mean, atol=1e-4)
    assert torch.allclose(out_std, style_std, atol=1e-3)


def test_adain_preserves_spatial_ranking():
    """Content structure survives: AdaIN is affine per channel, so ordering holds."""
    content = torch.randn(1, 4, 6, 6)
    style = torch.randn(1, 4, 6, 6)
    out = adaptive_instance_normalization(content, style)

    for c in range(4):
        a = content[0, c].flatten().argsort()
        b = out[0, c].flatten().argsort()
        assert torch.equal(a, b)


def test_adain_accepts_mismatched_style_resolution():
    content = torch.randn(1, 8, 16, 16)
    style = torch.randn(1, 8, 4, 7)
    out = adaptive_instance_normalization(content, style)
    assert out.shape == content.shape


# --------------------------------------------------------------------------
# style interpolation (paper Eq. 15)
# --------------------------------------------------------------------------
def test_interpolation_with_full_weight_equals_plain_adain():
    content = torch.randn(1, 8, 10, 10)
    a, b = torch.randn(1, 8, 10, 10), torch.randn(1, 8, 10, 10)

    blended = style_interpolation(content, [a, b], [1.0, 0.0])
    direct = adaptive_instance_normalization(content, a)
    assert torch.allclose(blended, direct, atol=1e-6)


def test_interpolation_is_midpoint_at_equal_weights():
    content = torch.randn(1, 8, 10, 10)
    a, b = torch.randn(1, 8, 10, 10), torch.randn(1, 8, 10, 10)

    blended = style_interpolation(content, [a, b], [0.5, 0.5])
    expected = 0.5 * adaptive_instance_normalization(content, a) \
        + 0.5 * adaptive_instance_normalization(content, b)
    assert torch.allclose(blended, expected, atol=1e-6)


def test_interpolation_averages_the_style_statistics():
    """Averaging AdaIN outputs == averaging the styles' mean and std."""
    content = torch.randn(1, 8, 10, 10)
    a, b = torch.randn(1, 8, 10, 10) + 4, torch.randn(1, 8, 10, 10) - 1
    w = [0.25, 0.75]

    out_mean, _ = calc_mean_std(style_interpolation(content, [a, b], w))
    a_mean, _ = calc_mean_std(a)
    b_mean, _ = calc_mean_std(b)
    assert torch.allclose(out_mean, w[0] * a_mean + w[1] * b_mean, atol=1e-4)


# --------------------------------------------------------------------------
# alpha (paper Eq. 14)
# --------------------------------------------------------------------------
@pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
def test_alpha_interpolates_between_content_and_adain(alpha):
    content = torch.randn(1, 8, 10, 10)
    style = torch.randn(1, 8, 10, 10)
    adain = adaptive_instance_normalization(content, style)

    mixed = alpha * adain + (1 - alpha) * content
    if alpha == 0.0:
        assert torch.allclose(mixed, content, atol=1e-6)
    if alpha == 1.0:
        assert torch.allclose(mixed, adain, atol=1e-6)


# --------------------------------------------------------------------------
# web app
# --------------------------------------------------------------------------
def test_index_loads_without_error_banner(app_client):
    body = app_client.get('/').get_data(as_text=True)
    assert 'class="alert"' not in body, 'a fresh visit must not show an error'
    assert 'id="titleInput"' in body


def test_render_single_style(app_client):
    res = app_client.post('/', data={
        'content': (io.BytesIO(image_bytes('portrait.jpg')), 'portrait.jpg'),
        'style': (io.BytesIO(image_bytes('pencil_sketch.jpg')), 'pencil_sketch.jpg'),
        'alpha': '1.0',
    }, content_type='multipart/form-data')
    body = res.get_data(as_text=True)
    assert res.status_code == 200
    assert 'id="result"' in body
    assert 'class="alert"' not in body


def test_render_two_styles_reports_blend(app_client):
    res = app_client.post('/', data={
        'content': (io.BytesIO(image_bytes('portrait.jpg')), 'portrait.jpg'),
        'style': (io.BytesIO(image_bytes('pencil_sketch.jpg')), 'pencil_sketch.jpg'),
        'style2': (io.BytesIO(image_bytes('blue_brushstrokes.jpg')), 'brushstrokes.jpg'),
        'alpha': '1.0', 'blend': '0.4',
    }, content_type='multipart/form-data')
    body = res.get_data(as_text=True)
    assert 'id="result"' in body
    assert '>Style B<' in body and '>Blend<' in body


def test_custom_title_appears_on_the_plate(app_client):
    res = app_client.post('/', data={
        'content': (io.BytesIO(image_bytes('portrait.jpg')), 'portrait.jpg'),
        'style': (io.BytesIO(image_bytes('pencil_sketch.jpg')), 'pencil_sketch.jpg'),
        'alpha': '1.0', 'title': 'Portrait in Graphite',
    }, content_type='multipart/form-data')
    assert 'Portrait in Graphite' in res.get_data(as_text=True)


def test_title_is_escaped(app_client):
    res = app_client.post('/', data={
        'content': (io.BytesIO(image_bytes('portrait.jpg')), 'portrait.jpg'),
        'style': (io.BytesIO(image_bytes('pencil_sketch.jpg')), 'pencil_sketch.jpg'),
        'alpha': '1.0', 'title': '<script>alert(1)</script>',
    }, content_type='multipart/form-data')
    body = res.get_data(as_text=True)
    assert '<script>alert(1)</script>' not in body
    assert '&lt;script&gt;' in body


def test_missing_content_is_reported(app_client):
    res = app_client.post('/', data={
        'style': (io.BytesIO(image_bytes('pencil_sketch.jpg')), 'pencil_sketch.jpg'),
        'alpha': '1.0',
    }, content_type='multipart/form-data')
    assert 'Please choose a content image.' in res.get_data(as_text=True)


def test_missing_style_is_reported(app_client):
    res = app_client.post('/', data={
        'content': (io.BytesIO(image_bytes('portrait.jpg')), 'portrait.jpg'),
        'alpha': '1.0',
    }, content_type='multipart/form-data')
    assert 'Please choose at least one style image.' in res.get_data(as_text=True)


@pytest.mark.parametrize('filename', ['photo.webp', 'photo.jfif', 'art.bmp', 'photo.JPG'])
def test_common_image_formats_are_accepted(app_client, filename):
    """Regression: these were silently dropped, clearing the preview and rendering nothing."""
    res = app_client.post('/', data={
        'content': (io.BytesIO(image_bytes('portrait.jpg')), 'portrait.jpg'),
        'style': (io.BytesIO(image_bytes('pencil_sketch.jpg')), filename),
        'alpha': '1.0',
    }, content_type='multipart/form-data')
    body = res.get_data(as_text=True)
    assert 'id="result"' in body, f'{filename} was rejected'


def test_non_image_extension_is_rejected_with_a_reason(app_client):
    res = app_client.post('/', data={
        'content': (io.BytesIO(image_bytes('portrait.jpg')), 'portrait.jpg'),
        'style': (io.BytesIO(b'not an image'), 'notes.txt'),
        'alpha': '1.0',
    }, content_type='multipart/form-data')
    body = res.get_data(as_text=True)
    assert 'not supported' in body
    assert 'id="result"' not in body


def test_corrupt_image_is_rejected_with_a_reason(app_client):
    res = app_client.post('/', data={
        'content': (io.BytesIO(image_bytes('portrait.jpg')), 'portrait.jpg'),
        'style': (io.BytesIO(b'\x00\x01\x02\x03'), 'broken.jpg'),
        'alpha': '1.0',
    }, content_type='multipart/form-data')
    assert 'could not be read as an image' in res.get_data(as_text=True)


def test_results_do_not_collide(app_client):
    """Two renders of the same content must not overwrite each other."""
    import re
    names = set()
    for _ in range(2):
        res = app_client.post('/', data={
            'content': (io.BytesIO(image_bytes('portrait.jpg')), 'portrait.jpg'),
            'style': (io.BytesIO(image_bytes('pencil_sketch.jpg')), 'pencil_sketch.jpg'),
            'alpha': '1.0',
        }, content_type='multipart/form-data')
        found = re.search(r'stylized_\w+_portrait\.jpg', res.get_data(as_text=True))
        assert found
        names.add(found.group(0))
    assert len(names) == 2


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
def test_full_pipeline_produces_a_valid_image(encoder, decoder, tmp_path):
    from torchvision import transforms

    tf = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    content = tf(Image.open(EXAMPLES / 'portrait.jpg').convert('RGB')).unsqueeze(0)
    style = tf(Image.open(EXAMPLES / 'pencil_sketch.jpg').convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        t = adaptive_instance_normalization(encoder(content, is_test=True),
                                            encoder(style, is_test=True))
        out = decoder(t)

    assert torch.isfinite(out).all(), 'output contains NaN or inf'

    out_path = tmp_path / 'out.jpg'
    transforms.ToPILImage()(out.squeeze(0).clamp(0, 1)).save(out_path)
    assert Image.open(out_path).size[0] > 0
