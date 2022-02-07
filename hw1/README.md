# Face Verification on the Asian Politician Dataset (APD)
高等多媒體資訊分析與檢索 | Advanced Topics in Multimedia Analysis and Indexing | NTU  | 徐宏民教授 | 2021 Spring | https://winstonhsu.info/ammai-21s/

## Goal
Solve the face verification problem using deep learning skill with various cost function. Train the model on the self-collected [Asian Politician Dataset (APD)](https://drive.google.com/file/d/1-GUAGJcSoiBaR8ecylUlJPvDJtqxCDLE/view?usp=sharing) and test its performance under two different settings: **Closed-set** and **Open-set**.

## Dataset Preview

<img src=https://user-images.githubusercontent.com/57071722/152835212-4508a127-448a-4dd5-99f2-a3c514ade615.jpg width="50%">

## Result
| Model                               | Closed-set Accuracy      | Open-set Accuracy  |
|:-----------------------------------:|:------------------------:|:------------------:|
| Baseline (AM-Softmax / Supervised)  | 71.8%                    | 80.9%              |
| Variant (AM-Softmax / Unsupervised) | 72.3%                    | 81.1%              |

## Method

### Baseline
我使用AM-Softmax作為baseline model的loss function，得到的結果為，closed-set accuracy 71.8%，open-set accuracy 80.9%。其中open-set accuracy反而比closed-set accuracy還要高的原可能是 closed-set 的資料沒有被清理過，因此其中包含多人的照片占蠻大一部份，如此一來在做 preprocessing 的時候，有可能 face detection選出的其中一人不是實際ground truth所指的該人。不過我也無法由 verification的 0, 1 ground truth去準確反推實際是希望比較多人中的哪一位，因此最終並沒有去清dataset，避免產生更多人為偏見造成的不準確性。

### Variant
依照baseline model，我再額外增加unsupervised learning的部分，使用到optional unlabeled image set in open set在訓練過程中。我所採用的unsupervised learning是將unlabeled data中的每一人都視不同人，再與APD中有labeled的資料做混合，並交由loss function做到intra-class compactness and inter-class separability。最終結果，closed-set accuracy 72.3%，open-set accuracy 81.1%，表現最好。
