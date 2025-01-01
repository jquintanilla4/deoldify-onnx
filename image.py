import os
import cv2
# import numpy as np
import onnxruntime as rt

rt.set_default_logger_severity(3)

def get_result_path(source_path):
    """Generate result path in the same directory as source image"""
    directory = os.path.dirname(source_path)
    filename = os.path.basename(source_path)
    name, ext = os.path.splitext(filename)
    return os.path.join(directory, f"{name}_colorized{ext}")


def main():
    # Get image source path from user input
    source_path = input("Enter the path to the image: ").strip()
    
    # Validate render factor
    try:
        render_factor = int(input("Render factor (recommended 8, must be divisible by 32): ").strip() or "8")
        if render_factor <= 0:
            print("Error: Render factor must be positive.")
            return
    except ValueError:
        print("Error: Please enter a valid number for render factor.")
        return

    # Remove quotes if present
    source_path = source_path.replace('"', '').replace("'", '').replace('`', '')

    if not os.path.isfile(source_path):
        print(f"Error: '{source_path}' is not a valid file.")
        return

    # Get export path
    export_path = get_result_path(source_path)
    
    '''
    The render factor determines the resolution at which the image is rendered for inference.
    When set at a low value, the process is faster and the colors tend to be more vibrant
    but the results are less stable.
    original torch model accepts input divisible by 16
    ONNX models currently accept only divisible by 32  
    '''
    multi_render_factor = render_factor * 32

    # OLD ONNX MODELS - you cannot set render_factor
    #from color.deoldify_fp16 import DEOLDIFY
    #colorizer = DEOLDIFY(model_path="color/deoldify_fp16.onnx", device="cpu")
    #from color.deoldify import DEOLDIFY
    #colorizer = DEOLDIFY(model_path="color/deoldify.onnx", device="cuda")

    # NEW ONNX MODELS - render_factor - dynamic axes input:
    # from color.deoldify_fp16 import DEOLDIFY
    # colorizer = DEOLDIFY(model_path="color/ColorizeArtistic_dyn_fp16.onnx", device="cuda")
    
    # Initialize colorizer
    from color.deoldify import DEOLDIFY
    colorizer = DEOLDIFY(model_path="color/ColorizeArtistic_dyn.onnx", device="cuda")

    # Process image with error handling
    try:
        image = cv2.imread(source_path)
        if image is None:
            print(f"Error: Unable to read image at '{source_path}'")
            return
            
        colorized = colorizer.colorize(image, multi_render_factor)
        cv2.imwrite(export_path, colorized)
        print(f"Colorized image saved to: {export_path}")
    except Exception as e:
        print(f"Error during image processing: {str(e)}")
        return


if __name__ == "__main__":
    main() 
