# Image-Captioning
This repository contains a transformer-based model for generating captions for images. The model has been trained on the COCO dataset, which consists of 120,000 images. The encoder component of the model uses ViT-B @224, which was pre-trained on the ImageNet22k dataset. The decoder component is a GPT-style model that is initialized with extracted image feature embeddings.

To improve the quality of the generated captions, beam search has been implemented.