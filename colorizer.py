import gradio as gr
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from model import ColorizationNet

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorizationNet()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Define the Gradio function
def colorize_image_gradio(image):
    # Preprocess image
    image = image.convert("L")  # Convert to grayscale
    original_size = image.size

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # [1, 1, 64, 64]

    # Predict
    with torch.no_grad():
        output = model(img_tensor)[0].cpu()  # [3, 64, 64]

    # Convert to NumPy image
    output_np = np.transpose(output.numpy(), (1, 2, 0))  # [64, 64, 3]
    output_np = np.clip(output_np, 0, 1)
    output_img = Image.fromarray((output_np * 255).astype(np.uint8))
    output_img = output_img.resize(original_size)

    return output_img

# Create Gradio interface
gr.Interface(
    fn=colorize_image_gradio,
    inputs=gr.Image(type="pil", label="Upload Grayscale Image"),
    outputs=gr.Image(label="Colorized Output"),
    title="Grayscale Image Colorizer"
).launch()
