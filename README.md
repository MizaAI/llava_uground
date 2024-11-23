# llava_uground

- [🏠Homepage](https://osu-nlp-group.github.io/UGround)
- [📖Paper](https://arxiv.org/abs/2410.05243)
- [😊Model Weights](https://huggingface.co/osunlp/UGround)
- [😊Live Demo](https://huggingface.co/spaces/orby-osu/UGround) (Try it out yourself!)
- [Main Code Repo](https://github.com/OSU-NLP-Group/UGround)

Note: This is the codebase to inference and train [UGround-v1](https://github.com/OSU-NLP-Group/UGround). It is modified from [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA), with changes mainly on image processing. We may use other codebases or architectures for later versions.

## Install

1. Create a python environment


```bash
conda create -n llava_uground python=3.11 -y
conda activate llava_uground
pip install --upgrade pip  # enable PEP 660 support
```


2. Install the dependencies

```bash
pip install -r requirements.txt
```

## Inference

```bash
python single_infer.py <model_selector> <image_path> <image_query>

eg:
python single_infer.py uground ../miza/datasets/test/calculator.png 'plus button'
```
Output of single_infer is stored in the image directory with query as suffix.

