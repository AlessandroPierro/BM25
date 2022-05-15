# BM25 with pseudo-relevance feedback

This repository contains the code for my final project of "Information Retrieval".

## Installation

Install the required Python libraries

```bash
pip install -qr requirements.txt
```

Download NLTK corpora (the ```reuters``` corpus is needed only to run the demo)

```python
import nltk
nltk.download('punkt')
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('omw-1.4')
nltk.download('reuters')
```
## Usage

After installing the required libraries, you can run the demo:

```python
python demo.py
```

## Author

- [Alessandro Pierro](https://github.com/AlessandroPierro)

## License

This project is licensed under the [MIT license](LICENSE).

## References

<a id="1">[1]</a> 
Stephen Robertson and Hugo Zaragoza - 
*The Probabilistic Relevance Framework: BM25 and Beyond*,
Foundations and Trends in Information Retrieval, 2009

<a id="2">[1]</a> 
Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze - 
*Introduction to Information Retrieval*,
Cambridge University Press. 2008