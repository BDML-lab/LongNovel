
<p align="center">
  <a href="--"><img src="https://img.shields.io/badge/arXiv-Paper-red"></a>
  <a href="https://github.com/BDML-lab/LongNovel/"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://huggingface.co/datasets/SII-BDML/LongNovel/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Benchmark-yellow"></a>
</p>

---
## Overview

While modern LLMs feature expanded context windows, hallucination remains a critical bottleneck—especially in long-context summarization. **LongNovel** is a multi-scale, bilingual (Chinese and English) benchmark specifically designed to study and detect hallucinations in long narrative texts.

### Why Novels?
Unlike short news articles or academic papers, long novels feature dense, intrinsic narrative structures, intricate event descriptions, and complex dialogues. This makes them the ideal testing ground for tracking how hallucinations evolve as context length scales.

### Key Features & Contributions

* **Bilingual & Multi-Scale Dataset:** Built using 29 Chinese novels (ranging from 16k to 100k tokens) alongside chapter-level English data from the BookSum dataset.
* **Granular Taxonomy:** Defines **8 distinct hallucination types** to categorize model errors precisely.

<div style="text-align: center;">
  <img src="assets/framework.jpg" width="1000">
</div>

## Updata

- `2026-06` We release LongNovel benchmark.

## Contents

- [Overview](#overview)
- [Updata](#updata)
- [Dataset](#dataset)
- [Results](#results)
- [License](#license)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## Dataset
### Load Data
You can download and load the LongNovel data through [this link](https://huggingface.co/datasets/SII-BDML/LongNovel) :

```Python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('SII-BDML/LongNovel', split='test')
```
### Data Format

The data format in **LongNovel** is structured as follows:

```json
{
    "id": "Unique identifier for each piece of data",
    "language": "The language of the data, which can be 'zh' for Chinese or 'en' for English",
    "article": "The original text corresponding to the summary to be detected",
    "summary": "The summary text that needs to be checked for hallucinations",
    "instruction": "The prompt or instruction used during the hallucination detection process",
    "input": "The complete string fed into the model, which contains both the article and the summary",
    "output": "The hallucination detection result"
}
```



## Results

We introduce LongNovel, a multilingual long-context dataset for hallucination detection in novels, based on human-annotated summaries. It comprises four subsets ranging from 16k to 100k tokens. Our extensive experiments on LongNovel reveal that current large language models still lack sufficient capability in long-context hallucination detection tasks. We hope that LongNovel will provide useful insights for future research in this field.

<p align="center">
  <img src="assets/Result.png" alt="Result" width="100%">
</p>

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgement

We would like to express our gratitude to the annotators from iQIYI for their high-quality manual labeling and correction. We also thank iQIYI for providing the GPU resources that supported this work. Furthermore, we thank the creators and maintainers of the BookSum dataset. Our work utilizes the processed version from the [ubaada/booksum-complete-cleaned](https://huggingface.co/datasets/ubaada/booksum-complete-cleaned) repository. We deeply appreciate both the original BookSum authors and the community contributors for making these valuable resources available.

## Citation

If you find our benchmark and code useful, please consider citing our work:

```bibtex
The ArXiv link will be added soon.
```




