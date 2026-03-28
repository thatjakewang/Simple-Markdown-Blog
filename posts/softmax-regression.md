---
title: 什麼是多項邏輯回歸（Softmax Regression）？
date: 2026-03-28
category: 機器學習
description: "Softmax Regression（又稱多項邏輯回歸，Multinomial Logistic Regression）是邏輯回歸的延伸模型。若把一般的邏輯回歸想成「是非題」，那麼 Softmax Regression 就像是「單選題」，專門用來處理三個類別以上的分類問題。"
---

## 什麼是 Softmax Regression 多項邏輯回歸？
Softmax Regression（又稱多項邏輯回歸，Multinomial Logistic Regression）是邏輯回歸的延伸模型。若把一般的邏輯回歸想成「是非題」，那麼 Softmax Regression 就像是「單選題」，專門用來處理三個類別以上的分類問題。

在現實世界中，多數問題都不是單純的二元判斷，因此 Softmax Regression 在實務上的應用其實更加廣泛。舉例來說，在車牌辨識系統中，模型需要從影像中判斷出每一個字元是數字還是英文字母。這時候模型並不是只回答「是不是某個數字」，而是同時評估所有可能的選項，例如 0–9 或 A–Z，並給出各自的機率。最終再選擇機率最高的結果，作為系統的判斷依據。

當攝影機拍到一個模糊的字元時，Softmax 會輸出類似「這有 85% 的機率是 8，10% 的機率是 B，其餘機率分散在其他可能」這樣的結果。這種「機率分佈」的表達方式，不僅讓模型可以做出預測，也讓系統能夠評估信心程度，進一步決定是否需要進行二次辨識或人工校驗。

## Softmax 在做什麼？

從本質上來看，Softmax 的工作其實很單純：它負責把模型輸出的「分數」轉換成「機率」。在模型內部，每個類別都會得到一個分數（通常稱為 logits），這些分數本身沒有明確的意義，也不能直接解讀為機率。

Softmax 的作用，就是把這些分數轉換成一組介於 0 到 1 之間的數值，並且讓所有數值加總為 1。換句話說，它把「模型覺得哪個比較像」這種相對比較，轉換成「每個類別的機率是多少」這種可解釋的結果。

![Softmax 公式](/static/images/softmax-formula.png)

- **σ** 是 Softmax 函數  
- **z** 是 logits 向量  
- **K** 是總類別數  
- **i** 是目前要計算的類別

這個公式的關鍵在於指數函數（exponential）。當某個類別的分數稍微高一點時，經過指數轉換後，差距會被放大，使得最有可能的類別更加突出，進而提高分類的穩定性。即使兩個 logits 只差 0.1，經過 Softmax 後機率差距也可能變得非常明顯，這就是為什麼 Softmax 特別適合用在分類任務。

![logits vs Softmax](/static/images/logits-score-and-softmax-probability.png)

原本 logits 只差一點，經過 Softmax 後，最可能的類別 A 瞬間拉開到 82% 以上，差異變得非常明顯！

## 訓練時如何學習？
光有 Softmax 還不夠，模型還需要「知道自己錯得有多離譜」才能不斷進步。這時候最常用的損失函數就是 **Cross-Entropy Loss**（交叉熵損失）。

簡單來說，Cross-Entropy 會比較「模型輸出的機率分佈」跟「真實答案（One-Hot 標籤）」之間的差距。差距越大，損失值越高；差距越小，損失值越低。訓練過程中，模型會使用[梯度下降](https://jake.tw/gradient-descent/)不斷調整權重，讓這個損失值越來越小。

這樣一來，Softmax 不只是「把分數轉成機率」，還能讓整個模型透過反向傳播（Backpropagation）有效學習。

## 模型是如何做出預測的？
在實際運作上，Softmax Regression 並不是一個單獨的步驟，而是一整個流程的一部分。首先，模型會對輸入資料進行線性轉換，也就是透過權重與偏差計算出每個類別的分數。這一步與線性回歸非常類似，本質上是在做特徵加權。

接著，這些分數會被送入 Softmax 函數，轉換成一組機率分佈。這時候，每一個類別都會有一個對應的機率值，而且所有機率加總為 1，因此可以直接解讀為模型的信心程度。

最後，模型會選擇機率最高的那一個類別，作為最終的預測結果（這個動作稱為 Argmax）。整個過程看似簡單，但背後其實是透過大量標註資料、Cross-Entropy Loss 和優化器，反覆訓練出來的權重，才能在複雜的情境中做出合理判斷。

以下是用 PyTorch 實現的完整 Softmax 流程（只需 4 行核心程式碼）：

```python
import torch
import torch.nn.functional as F

# 模型輸出的原始分數
logits = torch.tensor([3.0, 1.0, 0.5])
# 轉成機率
probs = F.softmax(logits, dim=0)
print(probs)                    
# tensor([0.8214, 0.1112, 0.0674])                

# 選擇機率最高的類別
predicted_class = torch.argmax(probs)
# 輸出 0（代表類別 A）
print(predicted_class.item())
```
在實際寫程式時，通常會先減去 logits 中的最大值來確保計算安全，避免指數函數 overflow：

```python
logits = logits - logits.max()
probs = F.softmax(logits, dim=0)
```

這是 PyTorch 內部也會自動處理的技巧。


## Softmax 與 Logistic Regression 的關係
從結構上來看，Softmax Regression 可以被視為 Logistic Regression 的自然延伸。Logistic Regression 主要處理二分類問題，透過 Sigmoid 函數輸出單一機率；而 Softmax 則是將這個概念擴展到多個類別，同時輸出一整組機率分佈。

也因此，在實務上可以把 Softmax 想成「多分類版本的 Logistic Regression」。當問題從「是或否」變成「多選一」時，就會自然地從 Sigmoid 轉向 Softmax。

Softmax 假設每個樣本只能屬於一個類別（互斥）。如果一個樣本可能同時屬於多個類別（例如一張照片同時有貓和狗），就應該改用多個 Sigmoid（多標籤分類），而不是 Softmax。

## 實務應用
Softmax Regression 幾乎出現在所有需要分類的場景中。除了車牌辨識之外，像是影像分類（辨識照片中的物體）、手寫數字辨識（MNIST）、文件分類（垃圾郵件判斷）等，背後都可以看到 Softmax 的影子。

更重要的是，在現代深度學習模型中，Softmax 通常會出現在神經網路的最後一層，作為輸出層的標準配置。換句話說，只要是分類問題，你幾乎都會遇到它。

## 結論
Softmax Regression 的核心價值，在於它能夠把模型的輸出轉換成「可解釋的機率分佈」，並且自然地應用在多分類問題中。透過這樣的設計，模型不僅能做出預測，還能表達自身的信心程度。

如果用一句話來總結，它就是一個會「幫所有選項打分數，並轉換成機率後做出最佳選擇」的模型。而這個簡單但強大的機制，正是現代分類模型的基礎。