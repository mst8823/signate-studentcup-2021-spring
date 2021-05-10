# Student Cup 2021 Spring
## 6th plath solution
- コンペURL: https://signate.jp/competitions/449
- public 1st, private 6th

0. 実行

    - Google Colaboratory (無料版)
    - Google Drive > MyDrive > studentcup-2021-spring フォルダを作成

    - directory structure
        ```
        studentcup-2021-spring
        ├── input
        │   ├── genre_labels.csv
        │   ├── sample_submit.csv
        │   ├── test.csv
        │   └── train.csv
        └── notebooks
            ├── 1_1_preprocess.ipynb
            ├── 1_2_train.ipynb
            ├── 1_3_predict.ipynb
            ├── 2_1_preprocess.ipynb
            ├── 2_2_train.ipynb
            └── 2_3_predict.ipynb
        ```
    - `1_1_preprocess.ipynb` -> `1_2_train.ipynb` -> `1_3_predict.ipynb`     
    -> `2_1_preprocess.ipynb` -> `2_2_train.ipynb` -> `2_3_predict.ipynb`
    の順番で実行

1. feature enginnering
    1. LightGBM features
        - raw features : "poplurarity", "acousticness" などのそのままの値
        - binned features : popularity の10の位と1の位 (pop10, pop01)
        - tempo features : "tempo_low", "tempo_high", "high-low"
        - corss features : pop10 × region -> pop10region
        - CE features : "region", "pop10region" の count encode
        - OE features : "region", "pop10region" の ordinal encode
        - TE features : "region", "pop10region" の target encode
        - aggregational features :     
             key="region", "pop10region",    
             values=raw_features,     
             method=\["min", "mean", "max", max_min, "z-score", "var", "skew", kurt\]
        - num nan features : 欠損値の行和
    
    2. kNN features
        - nagiss さんの特徴量 : [フォーラム](https://signate.jp/competitions/449/discussions/knn-baseline-cv06256-lb06417)
        - TE features : "pop10region" の target encode
    
    3. MLP features
        - raw features
        - binned features
        - tempo features
        - corss features
        - CE features
        - TE features
        - OHE features : "region", "pop10region" の one hot encode
        - aggregational features :     
             key="region", "pop10region",    
             values=raw_features,     
             method=\["z-score"\]
    
    - popularity10 + region の追加とその集約特徴量，Target Encode 特徴量の追加が精度向上にかなり貢献した
    - "popularity"×"acousticness", "loudness"×"positiveness"などの掛け合わせ特徴量は精度向上に貢献しなかった．もしかしたら良い組み合わせがあったかもしれない

2. cv strategy 
    - 層化kfold (key="genre"+"region")

3. model
    ![2021-05-10-00-16-58](https://user-images.githubusercontent.com/64417843/117602853-9f3c5080-b18c-11eb-9bd6-c0ca00315a9e.png)
    - Fist Stage Prediction (10 seeds, 15 folds)
        1. lightGBM : class_weight="balanced" (逆ラベル頻度で重みづけ)
        2. kNN : n_neighbors=5, weights="distance"
        3. MLP : 3層からなるシンプルなモデル
        4. stacking : 1~3の出力を入力としてMLPでstacking
        5. weighted average : 1~4の出力に対し，4つの最適化した重みで加重平均

    - Pseudo Labeling (5 seeds, 10 folds)
        1. First Stage Prediction の予測値から，最大予測確率が0.7以上のテストデータのレコードを用いて学習データの水増し (sample size : 4046 -> 6624)
        2. Masked Pseudo Labeling
            - 水増しされたテストデータのレコードが学習用データにあるとき，同一レコードの予測値を隠して，同一レコードによる予測を防いで予測を行う    
            - 行っていることが正しいのかはわかんないです
            - 少し前のですが，kaggle GM akibaさんの [cross pseudo labeling](https://speakerdeck.com/iwiwi/kaggle-state-farm-distracted-driver-detection)を参考にしました
        3. first stage と同様に予測をしたものを最終サブミッション
    
    - pseudo labeling の効果
        - public score :  0.6889 -> 0.6890 (他のsubでは0.001~0.009程度の精度向上) 
        - private score : 0.6863 -> 0.7003

    - cv score, public LB には stacking と weighted average, pseudo labeling の効果があった

4. feature importance
    ![2021-05-09-23-04-28](https://user-images.githubusercontent.com/64417843/117602760-70be7580-b18c-11eb-9abc-ec9b7845ba83.png)

    - first stage, lightGBM の 特徴量重要度
    - target encode 特徴量の重要度が少し高く，leak気味な気がしたが CV, LBともTEを用いないものと比べて比例して向上だったので，大丈夫そうだと思い採用

5. confusion matrics
    - first stage の混合行列のプロット
    ![2021-05-09-23-05-10](https://user-images.githubusercontent.com/64417843/117602898-b67b3e00-b18c-11eb-9b0e-831c0f1a00c1.png)


