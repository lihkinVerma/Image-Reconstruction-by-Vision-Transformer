# Image Reconstruction using Enhanched Vision Transformer
This repository is for maintainig code developed for CSC2547 project - Image Reconstruction using Vision transformer for Tiny Imagenet.

### About
Removing noise from images is a challenging and fundamental problem in the field of computer vision. Images captured by modern cameras are inevitably degraded by noise which limits the accuracy of any quantitative measurements on those images. In this project, we propose a novel image reconstruction framework which can be used for tasks such as image denoising, deblurring or inpainting. The model proposed in this project is based on Vision Transformer (ViT) [1] that takes 2D images as input and outputs embeddings which can be used for reconstructing denoised images. We incorporate four additional optimization techniques in the framework to improve the model reconstruction capability, namely Locality Sensitive Attention (LSA) [2], Shifted Patch Tokenization (SPT) [2], Rotary Positional Embedding (RoPE) [3] and adversarial loss function inspired from Generative Adversarial Networks(GANs) [4]. LSA, SPT and RoPE enable the transformer to learn from the dataset more efficiently, while the adversarial loss function enhances the resolution of the reconstructed images. Based on our experiments, the proposed architecture outperforms the benchmark U-Net model by more than 3.5% structural similarity (SSIM) across various reconstruction tasks, including image denoising and inpainting. The proposed enhancements further show an improvement of ~5% SSIM over the benchmark for both tasks.

In this project, we propose a novel image reconstruction framework using ViT by enhancing various components of the ViT architecture. The inputs to a standard ViT are the pixel patches, called tokens, generated from the original image. These tokens are concatenated with the positional embeddings and fed to the transformer, after which attention is applied. In this project, the enhancements we propose to the ViT are as follows:
- Use the SPT techniqueto improve the tokenization process by generating overlapping tokens.
- Use the RoPE method to improve the way positions are encoded with the tokens.
- Enhance the attention mechanism to avoid smoothening of attention scores by using a learnable temperature employing the LSA technique.
- Use a discriminator to calculate the binary cross entropy loss of the reconstructed images to further improve the resolution of the reconstructed images.
We tested our architecture on two reconstruction tasks, image denoising and inpainting, and compared the results against the benchmark U-Net model and the baseline vanilla ViT. We also performed a comprehensive analysis of the various proposed enhancements. The proposed framework can be used for applications such as MRI.

![architecture](https://user-images.githubusercontent.com/22643549/163879188-b0371762-7560-4901-b8f8-8e5bf0df626f.png)

### References
1. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D.Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
2. S. H. Lee, S. Lee, and B. C. Song, “Vision transformer for small-size datasets,” arXiv preprint arXiv:2112.13492, 2021.
3. P. Jeevan and A. Sethi, “Vision xformers: Efficient attention for image classification,” arXiv preprint arXiv:2107.02239, 2021.
4. K. Lee, H. Chang, L. Jiang, H. Zhang, Z. Tu, and C. Liu, “Vitgan: Training gans with vision transformers,” arXiv preprint arXiv:2107.04589, 2021.

### Authors
1. Lydia Chau: lydia.chau@mail.utoronto.ca
2. Deepkamal K. Gill: deepkamal.gill@mail.utoronto.ca
3. Nikhil Verma: lih.verma@mail.utoronto.ca
