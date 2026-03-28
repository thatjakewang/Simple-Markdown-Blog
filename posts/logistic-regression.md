---
title: 什麼是邏輯回歸（Logistic Regression）？
date: 2026-03-28
category: 機器學習
description: "Logistic Regression（邏輯回歸）是機器學習中最經典的二分類模型。若把 Softmax Regression 想成「單選題」，那麼 Logistic Regression 就像是「是非題」，專門用來處理只有兩個類別的問題。它透過 **Sigmoid 函數** 把模型輸出的分數轉換成介於 0 到 1 之間的機率。"
---

## 什麼是 Logistic Regression 邏輯回歸？

Logistic Regression（邏輯回歸）是機器學習中最經典的二分類模型。若把 Softmax Regression 想成「單選題」，那麼 Logistic Regression 就像是「是非題」，專門用來處理只有兩個類別的問題。它透過 **Sigmoid 函數** 把模型輸出的分數轉換成介於 0 到 1 之間的機率。

在現實世界中，很多決策本質上就是「是或否」，因此 Logistic Regression 是實務上應用最廣泛的模型之一。舉例來說，在垃圾郵件過濾系統中，模型需要判斷一封郵件是「垃圾郵件」還是「正常郵件」。這時候模型並不是列出所有可能，而是直接給出「這是垃圾郵件的機率」。

當收到一封可疑郵件時，Logistic Regression 會輸出類似「這有 92% 的機率是垃圾郵件」的結果。這種機率式的表達方式，不僅讓模型可以做出預測，也讓系統能夠評估信心程度，進一步決定是否要直接移入垃圾桶或進行人工審核。

## Sigmoid 在做什麼？

從本質上來看，Sigmoid 的工作其實很單純：它負責把模型輸出的「分數」轉換成「介於 0 到 1 之間的機率」。在模型內部，輸入資料會先經過線性轉換得到一個分數（通常稱為 logit），這個分數本身沒有明確的意義。

Sigmoid 的作用，就是把這個分數壓縮到 0～1 之間，讓它可以直接被解讀為「正類的機率」。換句話說，它把「模型覺得有多像正類」這種相對比較，轉換成「這是正類的機率是多少」這種可解釋的結果。

![Sigmoid 公式](/static/images/sigmoid-formula.png)

- **σ** 是 Sigmoid 函數  
- **z** 是 logit 分數

這個公式的關鍵在於指數函數（exponential）。當 logit 分數越大，機率越接近 1；當 logit 分數越小，機率越接近 0。這種平滑的 S 型曲線，讓模型在邊界附近不會有太過突兀的跳躍。

## 訓練時如何學習？
光有 Sigmoid 還不夠，模型還需要「知道自己錯得有多離譜」才能不斷進步。這時候最常用的損失函數就是 **Cross-Entropy Loss**（交叉熵損失）。

簡單來說，Cross-Entropy 會比較「模型輸出的機率」跟「真實答案（0 或 1）」之間的差距。差距越大，損失值越高；差距越小，損失值越低。訓練過程中，模型會使用[梯度下降](https://jake.tw/gradient-descent/)不斷調整權重，讓這個損失值越來越小。

這樣一來，Sigmoid 不只是「把分數轉成機率」，還能讓整個模型透過反向傳播（Backpropagation）有效學習。

## 模型是如何做出預測的？

Logistic Regression 本質上是**線性回歸的延伸**。線性回歸的輸出是連續值（從 \(-\infty\) 到 \(+\infty\)），而我們用 Sigmoid 把這個無限範圍的分數壓縮到 \((0, 1)\) 之間，變成可解釋的機率。

因此，在高維空間中，Logistic Regression 其實就是在找一個**超平面（Decision Boundary）**，把兩個類別分隔開來。

首先，模型會對輸入資料進行線性轉換，也就是透過權重與偏差計算出 logit 分數。這一步與線性回歸非常類似，本質上是在做特徵加權。

接著，這個分數會被送入 Sigmoid 函數，轉換成一個介於 0 到 1 之間的機率值。這個機率值可以直接解讀為模型的信心程度（例如 0.92 代表 92% 可能是正類）。

最後，模型通常會設定一個門檻值（最常用的是 0.5），機率大於門檻值就預測為正類，小於則預測為負類。整個過程看似簡單，但背後其實是透過大量標註資料、Cross-Entropy Loss 和優化器，反覆訓練出來的權重，才能在複雜的情境中做出合理判斷。

以下是用 PyTorch 實現的完整 Logistic Regression 流程：

```python
import torch
import torch.nn.functional as F

# 模型輸出的原始分數（logit）
logit = torch.tensor([2.3])
# 轉成機率
prob = F.sigmoid(logit)
print(prob)                    
# tensor([0.9089])                

# 設定門檻值 0.5 做最終預測
predicted = (prob > 0.5).float()
# 輸出 1.0（代表正類）
print(predicted.item())
```

在實際寫程式時，雖然 Sigmoid 本身較不容易 overflow，但後續計算 Cross-Entropy 時經常會遇到 log(0) 的數值問題（導致 loss 變成 NaN）。

這也是為什麼 PyTorch 官方強烈推薦使用 BCEWithLogitsLoss 而不是「先 Sigmoid 再 BCELoss」的原因。BCEWithLogitsLoss 內部會自動使用 log-sum-exp trick 同時處理 Sigmoid + Cross-Entropy，避免中間產生不穩定的中間值，大幅提升數值穩定性與訓練速度。

## Logistic Regression 與 Softmax Regression 的關係

從結構上來看，Logistic Regression 可以被視為 [Softmax Regression 多項邏輯回歸](https://jake.tw/softmax-regression/) 的特例。當類別數只有兩個時，Softmax 實際上就會退化成 Sigmoid。

也因此，在實務上可以把 Logistic Regression 想成「二分類版本的 Softmax」。當問題從「多選一」簡化成「是或否」時，就會自然地從 Softmax 轉向 Sigmoid。

## 結論
Logistic Regression 的核心價值，在於它能夠把模型的輸出轉換成「可解釋的機率」，並且自然地應用在二分類問題中。透過這樣的設計，模型不僅能做出預測，還能表達自身的信心程度。

如果用一句話來總結，它就是一個會「幫輸入資料打一個分數，轉換成機率後做出是或否的最佳選擇」的模型。而這個簡單但強大的機制，正是理解 Softmax Regression 與現代分類模型的基礎。