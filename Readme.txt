Assignment 5 (Bonus) - Image Generation Using Diffusion Models
Anwar, MSDS25002
================================================================

FOLDER CONTENTS
---------------
MSDS25002_05.py          - Data loader, forward diffusion process, custom
                            loss, training loop, and image-generation
                            (sampling) function. Run this to train.
MSDS25002_05_model.py     - UNet denoising model architecture (predicts
                            noise epsilon given a noisy image and timestep).
MSDS25002_05_allCode.py   - Single file containing the code of both files
                            above, concatenated, as required by the
                            assignment spec.
test_single_sample.ipynb - Loads a trained checkpoint from saved_models/
                            and generates images from pure noise.
saved_models/             - Trained model checkpoint(s) (diffusion_model.pt).
outputs/                  - Generated figures: forward_process.png,
                            loss_curve.png, generated_samples.png.
Report.pdf                - Written report with results and discussion.
requirements.txt          - Python package versions used (CPU PyTorch).

NOTE: animal_data/ (the dataset) is NOT included in this submission, per
the assignment instructions ("DON'T RESUBMIT THE DATASETS PROVIDED").
To re-run training you must place the animal_data folder (with one
sub-folder per class, e.g. animal_data/Cat/, animal_data/Dog/, ...) in
this directory, or pass its path via --data_dir.

HOW TO RUN
----------
1. Create a Python environment and install dependencies:
       python -m venv venv
       venv\Scripts\activate          (Windows)
       pip install -r requirements.txt

2. Train the model (writes checkpoint to saved_models/ and figures to
   outputs/):
       python MSDS25002_05.py --data_dir animal_data ^
           --classes Cat Dog Bird Lion Tiger --images_per_class 20 ^
           --img_size 64 --timesteps 1000 --epochs 200 --batch_size 8 ^
           --lr 2e-4 --num_samples 8

   Command line arguments (all optional, defaults shown):
       --data_dir          path to the animal_data folder   (default: animal_data)
       --classes           list of class folder names        (default: Cat Dog Bird Lion Tiger)
       --images_per_class  images sampled per class           (default: 20)
       --img_size           output image resolution            (default: 64)
       --timesteps         diffusion steps T                  (default: 1000)
       --epochs            training epochs                     (default: 300)
       --batch_size        mini-batch size                     (default: 8)
       --lr                learning rate                       (default: 2e-4)
       --save_dir          where to write the model checkpoint  (default: saved_models)
       --output_dir        where to write result figures        (default: outputs)
       --num_samples       images to generate after training    (default: 8)
       --seed              random seed                          (default: 42)

3. To generate new samples from an already-trained checkpoint without
   retraining, open test_single_sample.ipynb in Jupyter and run all cells.
   It loads saved_models/diffusion_model.pt and runs the reverse diffusion
   process to produce new images.

RESULTS (from the included saved_models/diffusion_model.pt checkpoint)
-----------------------------------------------------------------------
- Trained for 200 epochs on 100 images (5 classes x 20 images each, 64x64).
- Training loss (custom noise-prediction MSE) dropped from ~0.70 at epoch 1
  to ~0.02-0.03 by epoch 100, then oscillated in that range through epoch 200
  (see outputs/loss_curve.png).
- Forward process reaches near-pure Gaussian noise by t~400-500 of 1000
  steps (see outputs/forward_process.png).
- Samples generated via the full reverse process (outputs/generated_samples.png)
  show coherent color/texture patches consistent with the trained animal
  classes, but not sharp recognizable animal shapes - expected given the
  very small dataset (100 images) and CPU-only training budget. See
  Report.pdf section 4-5 for a detailed discussion and a bug we hit and
  fixed (all-black outputs from reverse-process divergence).

IMPLEMENTATION NOTES
---------------------
- The forward process uses the closed-form DDPM reparameterization
  x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon, which is
  mathematically equivalent to repeatedly applying single-step Gaussian
  noise q(x_t | x_t-1) for t steps, but is computed in one shot. Noise is
  never added directly to the raw image; it is always scaled by the
  diffusion schedule (see GaussianDiffusion.q_sample in MSDS25002_05.py).
- The denoising model is a small UNet with sinusoidal timestep embeddings,
  GroupNorm + SiLU residual blocks, and skip connections between the
  downsampling and upsampling paths (MSDS25002_05_model.py).
- The loss is a manually written mean-squared-error between the predicted
  and true noise (noise_prediction_loss in MSDS25002_05.py), per the
  "custom loss function" requirement.
- Sampling follows DDPM Algorithm 2 (ancestral sampling): starting from
  x_T ~ N(0, I), the model iteratively predicts and removes noise for
  t = T-1, ..., 0.
