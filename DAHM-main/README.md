# DCDMM: Behavior Condition Diffusion Model for Multi-Modal Recommendation


## 📝 Environment

pip install -r requirements.txt

## 🚀 How to run the codes

The Netflix, Allrecipes and TikTok datasets are too large to include here, but you can download them from [GoogleDrive](https://drive.google.com/drive/folders/1AB1RsnU-ETmubJgWLpJrXd8TjaK_eTp0?usp=share_link).

Put your downloaded data (e.g. Netflix) under `data/` dir.

Run the code from within the `src/` directory.

- Netflix

```python
python main.py -d netflix -m NETFLIX5
```

- Allrecipes

```python
python main.py -d allrecipes -m ALLRECIPES5
```

- TikTok

```python
python main.py -d tiktok -m ATIKTOK5
```




