# Image Reconstruction using Enhanched Vision Transformer
This repository is for maintainig code developed for CSC2547 project - Image Reconstruction using Vision transformer for Tiny Imagenet.

### Project details
Removing noise from images is a challenging and fundamental problem in the field of computer vision. Images captured by modern cameras are inevitably degraded by noise which limits the accuracy of any quantitative measurements on those images. In this project, we propose a novel image reconstruction framework which can be used for tasks such as image denoising, deblurring or inpainting. The model proposed in this project is based on Vision Transformer (ViT) that takes 2D images as input and outputs embeddings which can be used for reconstruct- ing denoised images. We incorporate four additional optimization techniques in the framework to improve the model reconstruction capability, namely Locality Sensitive Attention (LSA), Shifted Patch Tokenization (SPT), Rotary Positional Embedding (RoPE) and adversarial loss function inspired from Generative Ad- versarial Networks(GANs). LSA, SPT and RoPE enable the transformer to learn from the dataset more efficiently, while the adversarial loss function enhances the resolution of the reconstructed images. Based on our experiments, the proposed architecture outperforms the benchmark U-Net model by more than 3.5% structural similarity (SSIM) across various reconstruction tasks, including image denoising and inpainting. The proposed enhancements further show an improvement of ~5% SSIM over the benchmark for both tasks.

### Vision Transformer for reconstruction task
In this project, we propose a novel image reconstruction framework using ViT by enhancing various components of the ViT architecture. The inputs to a standard ViT are the pixel patches, called tokens, generated from the original image. These tokens are concatenated with the positional embeddings and fed to the transformer, after which attention is applied. In this project, the enhancements we propose to the ViT are as follows:
- Use the SPT techniqueto improve the tokenization process by generating overlapping tokens.
- Use the RoPE method to improve the way positions are encoded with the tokens.
- Enhance the attention mechanism to avoid smoothening of attention scores by using a learnable temperature employing the LSA technique.
- Use a discriminator to calculate the binary cross entropy loss of the reconstructed images to further improve the resolution of the reconstructed images.
  We tested our architecture on two reconstruction tasks, image denoising and inpainting, and compared the results against the benchmark U-Net model and the baseline vanilla ViT. We also performed a comprehensive analysis of the various proposed enhancements. The proposed framework can be used for applications such as MRI.

![architecture](https://user-images.githubusercontent.com/22643549/163879188-b0371762-7560-4901-b8f8-8e5bf0df626f.png)

### Authors

1. Lydia Chau: lydia.chau@mail.utoronto.ca
2. Deepkamal K. Gill: deepkamal.gill@mail.utoronto.ca
3. Nikhil Verma: lih.verma@mail.utoronto.ca

### Practises followed while developing
General rules:
1. Use requirements.txt to mention any dependency in the project
2. Please mention any code if taken from some place. Put the mark in references at the bottom
3. Try experiments and make sure to git commit the code.

## Refernces
1. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020. 1, 2
2. S. H. Lee, S. Lee, and B. C. Song, “Vision transformer for small-size datasets,” arXiv preprint arXiv:2112.13492, 2021. 1, 2, 3.4
3. P. Jeevan and A. Sethi, “Vision xformers: Efficient attention for image classification,” arXiv preprint arXiv:2107.02239, 2021. 1, 3.3
4. A. H. Lone and A. N. Siddiqui, “Noise models in digital image processing,” Global Sci-Tech, vol. 10, no. 2, pp. 63–66, 2018. 2
5. C. Guillemot and O. Le Meur, “Image inpainting: Overview and recent advances,” IEEE signal processing magazine, vol. 31, no. 1, pp. 127–144, 2013. 2
6. C. Dong, C. C. Loy, K. He, and X. Tang, “Learning a deep convolutional network for image super-resolution,” in European conference on computer vision, pp. 184–199, Springer, 2014. 2
7. S. Lim, J. Kim, and W. Kim, “Deep spectral-spatial network for single image deblurring,” IEEE Signal Processing Letters, vol. 27, pp. 835–839, 2020. 2
8. S. Nah, T. Hyun Kim, and K. Mu Lee, “Deep multi-scale convolutional neural network for dynamic scene deblurring,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3883–3891, 2017. 2
9. Y. Thesia, M. Suthar, T. Pandya, and P. Thakkar, “Image denoising with self-adaptive multi-unet valve,” in Soft Computing for Problem Solving, pp. 647–659, Springer, 2021. 2
10. Y.Yuan,L.Huang,J.Guo,C.Zhang,X.Chen,andJ.Wang,“Ocnet:Objectcontextforsemantic segmentation,” International Journal of Computer Vision, vol. 129, no. 8, pp. 2375–2398, 2021. 2
11. B. Wu, C. Xu, X. Dai, A. Wan, P. Zhang, Z. Yan, M. Tomizuka, J. Gonzalez, K. Keutzer, and P. Vajda, “Visual transformers: Token-based image representation and processing for computer vision,” arXiv preprint arXiv:2006.03677, 2020. 2
12. Y. Chen, M. Rohrbach, Z. Yan, Y. Shuicheng, J. Feng, and Y. Kalantidis, “Graph-based global reasoning networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 433–442, 2019. 2
13. C. Tian, Y. Xu, Z. Li, W. Zuo, L. Fei, and H. Liu, “Attention-guided cnn for image denoising,” Neural Networks, vol. 124, pp. 117–129, 2020. 2
14. Z. Chen, D. Li, W. Fan, H. Guan, C. Wang, and J. Li, “Self-attention in reconstruction bias u-net for semantic segmentation of building rooftops in optical remote sensing images,” Remote Sensing, vol. 13, no. 13, p. 2524, 2021. 2
15. Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10012–10022, 2021. 2
16. O. Kupyn, V. Budzan, M. Mykhailych, D. Mishkin, and J. Matas, “Deblurgan: Blind motion deblurring using conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8183–8192, 2018. 2
9
17. J.-C. Lin, W.-L. Wei, T.-L. Liu, C.-C. J. Kuo, and M. Liao, “Tell me where it is still blurry: Adversarial blurred region mining and refining,” in Proceedings of the 27th ACM International Conference on Multimedia, pp. 702–710, 2019. 2
18. T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, A. Tao, J. Kautz, and B. Catanzaro, “High-resolution image synthesis and semantic manipulation with conditional gans,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8798–8807, 2018. 2
19. I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” Advances in neural information processing systems, vol. 27, 2014. 2, 3.5
20. K. Lee, H. Chang, L. Jiang, H. Zhang, Z. Tu, and C. Liu, “Vitgan: Training gans with vision transformers,” arXiv preprint arXiv:2107.04589, 2021. 3.5
21. R. Durall, S. Frolov, J. Hees, F. Raue, F.-J. Pfreundt, A. Dengel, and J. Keuper, “Combining transformer generators with convolutional discriminators,” in German Conference on Artificial Intelligence (Künstliche Intelligenz), pp. 67–79, Springer, 2021. 3.5
22. O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, “ImageNet Large Scale Visual Recognition Challenge,” International Journal of Computer Vision (IJCV), vol. 115, no. 3, pp. 211–252, 2015. 4.1
23. A. Hore and D. Ziou, “Image quality metrics: Psnr vs. ssim,” in 2010 20th international conference on pattern recognition, pp. 2366–2369, IEEE, 2010. 4.2
24. A. Rehman and Z. Wang, “Ssim-based non-local means image denoising,” in 2011 18th IEEE International Conference on Image Processing, pp. 217–220, IEEE, 2011. 4.2
