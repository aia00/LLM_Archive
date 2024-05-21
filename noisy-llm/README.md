# noisy-llm

Create an conda environment from the `environment.yml` file.
```
conda env create -f environment.yml
conda activate noise
```

Install the modified transformers library `transformers_noisy` to perform model inference with noise injected into input embeddings.
```
git clone https://github.com/haikangdeng/transformers_noisy.git
cd transformers_noisy
pip install -e .
```
