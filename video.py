import os
import sys
import cv2
# import numpy as np
import subprocess
import platform
from tqdm import tqdm
import onnxruntime as rt

rt.set_default_logger_severity(3) # Set logging level to 3 (INFO)

def init_onnx():
    rt.set_default_logger_severity(3)
    print("Available Providers:", rt.get_available_providers())
    print("CUDA Available:", 'CUDAExecutionProvider' in rt.get_available_providers())


def get_user_inputs():
    print("\n=== Video Colorization Settings ===")
    source = input("Enter path to source video: ").strip('"')
    
    # Generate output filename by adding "-colorized" before the extension
    base_path = os.path.splitext(source)[0]
    extension = os.path.splitext(source)[1]
    result = f"{base_path}-colorized{extension}"
    print(f"Output will be saved as: {result}")
    
    audio = input("Keep audio? (yes/no): ").lower().startswith('y')
    
    try: # Validate render factor
        render_factor = int(input("Render factor (recommended 8, must be divisible by 32): ").strip() or "8")
        if render_factor <= 0:
            print("Error: Render factor must be positive.")
            return
    except ValueError:
        print("Error: Please enter a valid number for render factor.")
        return
    
    return source, result, audio, render_factor


def process_video(source, result, audio, render_factor, colorizer):
    if not os.path.exists(source):
        print(f"Error: Source video file '{source}' not found!")
        sys.exit(1)

    video = cv2.VideoCapture(source)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))    
    fps = video.get(cv2.CAP_PROP_FPS)

    print(f"\nProcessing video: {w}x{h} at {fps}fps ({n_frames} frames)")

    output_path = 'temp.mp4' if audio else result
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))

    print("\nColorizing video frames...")
    for frame_idx in tqdm(range(n_frames)):
        ret, frame = video.read()
        if not ret:
            break
        result_frame = colorizer.colorize(frame, render_factor)
        writer.write(result_frame)

    writer.release()
    video.release()
    return output_path


def process_audio(source, result, temp_video):
    print("\nProcessing audio...")
    probe_command = f'ffprobe -i "{source}" -show_streams -select_streams a -loglevel error'
    has_audio = subprocess.call(probe_command, shell=platform.system() != 'Windows') == 0

    if has_audio:
        print("Merging audio with colorized video...")
        command = f'ffmpeg -y -vn -i "{source}" -an -i {temp_video} -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest "{result}"'
        subprocess.call(command, shell=platform.system() != 'Windows')
        os.remove(temp_video)
    else:
        print("No audio stream found in source video")
        os.rename(temp_video, result)


def main():
    # Initialize ONNX runtime
    init_onnx()

    # Get user inputs
    source, result, audio, render_factor = get_user_inputs()

    '''
    The render factor determines the resolution at which the image is rendered for inference.
    When set at a low value, the process is faster and the colors tend to be more vibrant
    but the results are less stable.
    Original torch model accepts input divisible by 16
    ONNX models currently accept only divisible by 32  
    '''
    multi_render_factor = render_factor * 32

    # OLD ONNX MODELS - you cannot set render_factor
    #from color.deoldify_fp16 import DEOLDIFY
    #colorizer = DEOLDIFY(model_path="color/deoldify_fp16.onnx", device="cpu")
    #from color.deoldify import DEOLDIFY
    #colorizer = DEOLDIFY(model_path="color/deoldify.onnx", device="cuda")

    # NEW ONNX MODELS - render_factor - dynamic axes input:
    # from color.deoldify import DEOLDIFY
    # colorizer = DEOLDIFY(model_path="color/ColorizeArtistic_dyn_fp16.onnx", device="cuda")
    
    # Initialize colorizer
    from color.deoldify import DEOLDIFY
    colorizer = DEOLDIFY(model_path="color/ColorizeArtistic_dyn.onnx", device="cuda")

    # Process video
    output_path = process_video(source, result, audio, multi_render_factor, colorizer)

    # Handle audio if needed
    if audio:
        process_audio(source, result, output_path)

    print(f"\nProcessing complete! Result saved to: {result}")


if __name__ == "__main__":
    main()
