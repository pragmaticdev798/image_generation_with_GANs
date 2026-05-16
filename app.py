"""
DCGAN Face Generation Web Application
Backend - Flask Application
Author: shreshth sharma 
Description: Main Flask app for face generation using trained DCGAN model
"""

import os
import json
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from datetime import datetime
import io
from pathlib import Path


from app.models.generator import Generator
from app.utils.image_utils import tensor_to_image, save_generated_image


app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')


app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
CHECKPOINT_DIR = 'checkpoints'
OUTPUT_DIR = 'outputs'
SAMPLES_DIR = 'generated_samples'


DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


generator = None
model_config = {
    'latent_size': 100,
    'num_channels': 3,
    'image_size': 64,
    'generator_channels': 64,
    'is_loaded': False,
    'checkpoint_path': None
}


def load_generator_model(checkpoint_path):
    """
    Load the trained generator model from checkpoint
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        generator model or None if loading fails
    """
    global generator, model_config
    
    # If a model is already loaded, and it's the same one, return it
    if generator is not None and model_config['checkpoint_path'] == checkpoint_path:
        print("✓ Generator model is already loaded.")
        return generator
    
    try:
        print(f"Loading generator from: {checkpoint_path}")
        
        # Initialize generator 
        gen = Generator(
            z_dim=model_config['latent_size'],
            channels_img=model_config['num_channels'],
            features_g=model_config['generator_channels']
        )
        
        # load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            # handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                # full checkpoint with training info
                gen.load_state_dict(checkpoint['generator_state_dict'])
            elif isinstance(checkpoint, dict) and 'generator' in checkpoint:
                # alternative format
                gen.load_state_dict(checkpoint['generator'])
            else:
                # direct state dict
                gen.load_state_dict(checkpoint)
            
            gen.to(DEVICE)
            gen.eval()  # set to evaluation mode
            
            model_config['is_loaded'] = True
            model_config['checkpoint_path'] = checkpoint_path
            generator = gen
            
            print("✓ Generator model loaded successfully!")
            return gen
        else:
            print(f"✗ Checkpoint file not found: {checkpoint_path}")
            model_config['is_loaded'] = False
            model_config['checkpoint_path'] = None
            generator = None
            return None
            
    except Exception as e:
        print(f"✗ Error loading generator: {str(e)}")
        model_config['is_loaded'] = False
        model_config['checkpoint_path'] = None
        generator = None
        return None


def unload_generator_model():
    """Unload the generator model and free up memory."""
    global generator, model_config
    if generator is not None:
        del generator
        generator = None
        model_config['is_loaded'] = False
        model_config['checkpoint_path'] = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()  # If using GPU
        print("✓ Generator model unloaded and memory freed.")


def generate_faces(num_images=5, seed=None):
    """
    Generate fake face images using the trained generator
    
    Args:
        num_images (int): Number of images to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of PIL Image objects
    """
    if generator is None:
        return []
    
    try:
        # set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        generated_images = []
        
        with torch.no_grad():
            for i in range(num_images):
                # generate random noise vector
                z = torch.randn(1, model_config['latent_size'], 1, 1, device=DEVICE)
                
                # generate image
                fake_image = generator(z)
                
                # Convert to PIL Image
                pil_image = tensor_to_image(fake_image)
                generated_images.append(pil_image)
        
        print(f"✓ Generated {num_images} face images successfully!")
        return generated_images
        
    except Exception as e:
        print(f"✗ Error generating images: {str(e)}")
        return []


#routes

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/generate_page')
def generate_page():
    """Render face generation page"""
    return render_template('generate.html')


@app.route('/gallery')
def gallery():
    """Render gallery page with training samples"""
    return render_template('gallery.html')


@app.route('/about')
def about():
    """Render about/model info page"""
    return render_template('about.html')


@app.route('/api/load-model', methods=['POST'])
def api_load_model():
    """
    API endpoint to load model from checkpoint
    
    Expected JSON:
        {
            "checkpoint_name": "dcgan_checkpoint_epoch_50.pth"
        }
    """
    try:
        data = request.get_json()
        checkpoint_name = data.get('checkpoint_name', '')
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        
        # model
        model = load_generator_model(checkpoint_path)
        
        if model is not None:
            return jsonify({
                'success': True,
                'message': f'Model loaded from {checkpoint_name}',
                'model_config': model_config
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to load model'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """
    API endpoint to generate face images
    
    Expected JSON:
        {
            "num_images": 5,
            "seed": null (optional)
        }
    """
    if generator is None:
        return jsonify({
            'success': False,
            'message': 'Model not loaded. Please load a checkpoint first.'
        }), 400
    
    try:
        data = request.get_json()
        num_images = int(data.get('num_images', 5))
        num_images = min(max(1, num_images), 20)  # Limit between 1-20
        
        seed = data.get('seed', None)
        if seed is not None:
            seed = int(seed)
        
        # generate images
        images = generate_faces(num_images, seed)
        
        if not images:
            return jsonify({
                'success': False,
                'message': 'Failed to generate images'
            }), 400
        
        # convert images to base64
        import base64
        image_data = []
        
        for idx, img in enumerate(images):
            # Save image
            filename = save_generated_image(img, OUTPUT_DIR)
            
            # convert to base64
            img_io = io.BytesIO()
            img.save(img_io, 'PNG', quality=95)
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode()
            
            image_data.append({
                'id': f'face_{int(datetime.now().timestamp() * 1000)}_{idx}',
                'data': f'data:image/png;base64,{img_base64}',
                'filename': filename
            })
        
        # Clean up memory
        del images
        import gc
        gc.collect()
        
        return jsonify({
            'success': True,
            'message': f'Generated {num_images} images successfully',
            'images': image_data,
            'count': len(image_data)
        })
        
    except Exception as e:
        print(f"Error in API generate: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/api/get-checkpoints', methods=['GET'])
def api_get_checkpoints():
    """Get list of available checkpoints"""
    try:
        if not os.path.exists(CHECKPOINT_DIR):
            return jsonify({'checkpoints': []})
        
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
        checkpoints.sort(reverse=True)  # Latest first
        
        return jsonify({
            'checkpoints': checkpoints,
            'count': len(checkpoints)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-samples', methods=['GET'])
def api_get_samples():
    """Get list of training sample images"""
    try:
        samples = []
        
        if os.path.exists(SAMPLES_DIR):
            for f in os.listdir(SAMPLES_DIR):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    samples.append(f)
        
        samples.sort()
        return jsonify({
            'samples': samples,
            'count': len(samples)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-model-info', methods=['GET'])
def api_get_model_info():
    """Get model configuration and info"""
    return jsonify({
        'model_config': model_config,
        'device': str(DEVICE),
        'is_loaded': model_config['is_loaded']
    })


@app.route('/api/download-image/<filename>', methods=['GET'])
def api_download_image(filename):
    """Download generated image"""
    try:
        # sanitize filename to prevent path traversal
        filename = os.path.basename(filename)
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/samples/<filename>')
def serve_sample(filename):
    """Serve sample image from generated_samples folder"""
    try:
        filepath = os.path.join(SAMPLES_DIR, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/png')
        else:
            return 'File not found', 404
    except Exception as e:
        return str(e), 500


# error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# application startup

if __name__ == '__main__':
    # create necessary directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("DCGAN Face Generation Web Application")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"Samples Dir: {SAMPLES_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
# application startup

if __name__ == '__main__':
    # create necessary directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("DCGAN Face Generation Web Application")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"Samples Dir: {SAMPLES_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # run flask app (development only)
    app.run(
        debug=False,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )
else:
    # production: create necessary directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
