import os
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField, StringField
from PIL import Image
from torchvision import transforms
from pathlib import Path
from uuid import uuid4

# Model and AdaIN operations
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, style_interpolation

# Resolve paths relative to this file so the app runs from any working directory
BASE_DIR = Path(__file__).resolve().parent


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-only-change-in-production')
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'static' / 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'jfif',
                                    'webp', 'bmp', 'gif', 'tif', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.context_processor
def inject_accept():
    # What the file picker offers, kept in step with ALLOWED_EXTENSIONS above.
    return {'ACCEPT': 'image/png,image/jpeg,image/webp,image/bmp,image/gif,image/tiff,.jfif'}

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    style2 = FileField('Second Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    style2_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    blend = FloatField('Blend', default=0.5)
    title = StringField('Title')
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder(str(BASE_DIR / 'vgg_normalised.pth')).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(BASE_DIR / 'experiment' / 'final_exp' / 'decoder_final.pth',
                                   map_location=device))

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_images, style_weights, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_images = [style_transform(s).unsqueeze(0).to(device) for s in style_images]

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = [encoder(s, is_test=True) for s in style_images]

        # One style -> plain AdaIN. Several styles -> interpolate between them (Eq. 15).
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


def save_upload(field, previous):
    # Store a newly uploaded file, or keep the one carried over from the last submit.
    # Returns (filename, error_message); exactly one of the two is set.
    if not (field.data and getattr(field.data, 'filename', '')):
        return (previous.data or None), None

    original = field.data.filename
    if not allowed_file(original):
        suffix = original.rsplit('.', 1)[-1].lower() if '.' in original else 'no extension'
        return None, (f'"{original}" is a .{suffix} file, which is not supported. '
                      f'Use PNG, JPEG, WebP, BMP, GIF or TIFF.')

    filename = secure_filename(original)
    if not filename:
        filename = f'upload_{uuid4().hex[:8]}.{original.rsplit(".", 1)[-1].lower()}'

    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    field.data.save(path)

    # Trust the file's contents, not its name.
    try:
        Image.open(path).convert('RGB')
    except Exception:
        return None, f'"{original}" could not be read as an image. It may be corrupt.'

    return filename, None


def read_float(field, default):
    try:
        return float(field.data)
    except (TypeError, ValueError):
        return default


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    style2_filename = None
    error = None

    if request.method == 'POST':
        if form.validate_on_submit():
            content_filename, content_error = save_upload(form.content, form.content_path)
            style_filename, style_error = save_upload(form.style, form.style_path)
            style2_filename, style2_error = save_upload(form.style2, form.style2_path)

            form.content_path.data = content_filename
            form.style_path.data = style_filename
            form.style2_path.data = style2_filename

            rejected = content_error or style_error or style2_error

            if rejected:
                error = rejected
            elif not content_filename:
                error = 'Please choose a content image.'
            elif not style_filename:
                error = 'Please choose at least one style image.'
            else:
                try:
                    content_image = Image.open(
                        os.path.join(app.config['UPLOAD_FOLDER'], content_filename)).convert('RGB')
                    style_images = [Image.open(
                        os.path.join(app.config['UPLOAD_FOLDER'], style_filename)).convert('RGB')]

                    alpha = read_float(form.alpha, 1.0)
                    blend = read_float(form.blend, 0.5)

                    if style2_filename:
                        style_images.append(Image.open(
                            os.path.join(app.config['UPLOAD_FOLDER'], style2_filename)).convert('RGB'))
                        style_weights = [1.0 - blend, blend]
                    else:
                        style_weights = [1.0]

                    stylized_image = style_transfer(content_image, style_images, style_weights,
                                                    encoder, decoder, alpha, device)

                    stem = Path(content_filename).stem
                    result_filename = f'stylized_{uuid4().hex[:8]}_{stem}.jpg'
                    save_image(stylized_image,
                               os.path.join(app.config['UPLOAD_FOLDER'], result_filename))

                    result_image = result_filename
                except Exception as e:
                    error = str(e)
        else:
            error = 'That submission could not be read. Please try again.'

    title = (form.title.data or '').strip()[:80]

    return render_template('index.html', form=form, result_image=result_image,
                           content_image=content_filename, style_image=style_filename,
                           style2_image=style2_filename, error=error, title=title)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)
