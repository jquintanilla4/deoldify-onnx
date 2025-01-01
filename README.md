# Deoldify-onnx

Option render factor only in commandline version

New models for use with render factor: 

https://drive.google.com/drive/folders/1bU9Zj7zGVEujIzvDTb1b9cyWU3s__WQR?usp=sharing

Simple image and video colorization using onnx converted deoldify model.

Easy to install. 
It can be run on CPU or Nvidia GPU

ffmpeg is required for video colorzation.

## For inference run:

Image:
```
python image.py
```
Video:
```
python video.py
```

Both Scripts are interactive. So you can follow along the simple question prompts.

## Image examples:
![colorizer1](https://github.com/instant-high/deoldify-onnx/assets/77229558/171642dd-9034-4ca7-8d29-c07c6e5e9f0a)


https://github.com/instant-high/deoldify-onnx/assets/77229558/3824e96d-fffc-494e-8ce1-193e6a77c8b6

https://github.com/instant-high/deoldify-onnx/assets/77229558/543e1dd1-27da-4c63-95a9-9c0696adea51


## Why Three Models?
There are now three models to choose from in DeOldify. Each of these has key strengths and weaknesses, and so have different use cases. Video is for video of course. But stable and artistic are both for images, and sometimes one will do images better than the other.

More details:

Artistic - This model achieves the highest quality results in image coloration, in terms of interesting details and vibrance. The most notable drawback however is that it's a bit of a pain to fiddle around with to get the best results (you have to adjust the rendering resolution or render_factor to achieve this). Additionally, the model does not do as well as stable in a few key common scenarios- nature scenes and portraits. The model uses a resnet34 backbone on a UNet with an emphasis on depth of layers on the decoder side. This model was trained with 5 critic pretrain/GAN cycle repeats via NoGAN, in addition to the initial generator/critic pretrain/GAN NoGAN training, at 192px. This adds up to a total of 32% of Imagenet data trained once (12.5 hours of direct GAN training).

Stable - This model achieves the best results with landscapes and portraits. Notably, it produces less "zombies"- where faces or limbs stay gray rather than being colored in properly. It generally has less weird miscolorations than artistic, but it's also less colorful in general. This model uses a resnet101 backbone on a UNet with an emphasis on width of layers on the decoder side. This model was trained with 3 critic pretrain/GAN cycle repeats via NoGAN, in addition to the initial generator/critic pretrain/GAN NoGAN training, at 192px. This adds up to a total of 7% of Imagenet data trained once (3 hours of direct GAN training).

Video - This model is optimized for smooth, consistent and flicker-free video. This would definitely be the least colorful of the three models, but it's honestly not too far off from "stable". The model is the same as "stable" in terms of architecture, but differs in training. It's trained for a mere 2.2% of Imagenet data once at 192px, using only the initial generator/critic pretrain/GAN NoGAN training (1 hour of direct GAN training).

Because the training of the artistic and stable models was done before the "inflection point" of NoGAN training described in "What is NoGAN???" was discovered, I believe this amount of training on them can be knocked down considerably. As far as I can tell, the models were stopped at "good points" that were well beyond where productive training was taking place. I'll be looking into this in the future.

Ideally, eventually these three models will be consolidated into one that has all these good desirable unified. I think there's a path there, but it's going


## Source Repos
Original Deoldify:
https://github.com/jantic/DeOldify

Forked Deoldify:
https://github.com/instant-high/deoldify-onnx