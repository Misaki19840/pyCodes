[オリジナルのSSD](https://github.com/pierluigiferrari/ssd_keras)を編集し、手だけを検出するように変更

## Note
- 元は２０クラスを検出するSSDの１クラスに対する出力をサブサンプリングし、転移学習で手を学習
- 画像サイズは300 x 300
- 1エポック目がベストで過学習になってしまう

![](./myCameraSSD_class1_hand.png)
