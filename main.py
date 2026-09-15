import os
import argparse
import uvicorn
from pathlib import Path
from Load_image import *
from config import *
from sharpening_model_pipeline import *
from postprocessing_model_pipeline import *

from server import create_app


def process_single_image(img_path: str, output_dir: str, device_type: str):
    print(f"-> Starting postprocessing: {img_path}")

    img_info = ImageInfo(img_path)
    loading_img = ImageLoader(img_info)

    normalized = Normalizer(loading_img.image, loading_img)
    normalized_img = normalized.scaling_01()

    yuv_im = ConvertRGB2YUV(normalized_img).converted_img_yuv

    if device_type == 'rpi5':
        sharpened = SharpenImageRPi(yuv_im, patch_size=512, min_overlap=64)()
    else:
        sharpened = SharpenImageStandard(yuv_im,min_overlap=96)()

    sharpened_rgb = ConvertYUV2RGB(normalized_img, sharpened)

    temp = Normalizer(sharpened_rgb.converted_img_rgb, loading_img)
    temp_im = temp.back_scaler()

    temp2 = Normalizer(temp_im, loading_img)
    temp_im2 = temp2.scaling_11()

    if device_type == 'rpi5':
        postprocessed = PostprocessingModelPipelineRpi(temp_im2, patch_size=512, min_overlap=128)
        postprocessed_im = postprocessed()
    else:
        postprocessed = PostprocessingModelPipeline(temp_im2, patch_size=512, min_overlap=256, batch_size=16)
        postprocessed_im = postprocessed()

    final = Normalizer(postprocessed_im, loading_img)
    final_im = final.back_scaler()

    SaveImage(img_info, output_dir).save_image(final_im)
    print(f"<- Finished: {img_info.img_name}")


def run_local_batch(input_dir: str, output_dir: str, device_type: str):
    input_path = Path(input_dir)
    extensions = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".tif", ".jpg", ".png"}

    files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    if not files:
        print(f"No images for postprocess {input_dir}")
        return

    for file in files:
        process_single_image(str(file), output_dir, device_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "local", "server"], default="auto")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    sys_info = SystemInfo()()
    paths = PathInfo()

    device = sys_info['device']
    mode = args.mode

    if mode == "auto":
        mode = "server" if device == 'rpi5' else "local"


    print(f"[*] System: {sys_info['system']} | RAM: {sys_info['RAM_gb']} GB")

    os.makedirs(paths.input_dir, exist_ok=True)
    os.makedirs(paths.output_dir, exist_ok=True)

    if mode == "server":
        print(f"[*] Server on port {args.port}...")

        def process_fn(img_p, out_p):
            process_single_image(img_p, out_p, device)

        app = create_app(process_fn)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        print(f"[*] Local mode {paths.working_dir}")
        run_local_batch(paths.input_dir, paths.output_dir, device)


if __name__ == "__main__":
    main()