# 078 Kobe 2017 Fashion Recommend API Server Demo

[078](https://078kobe.jp/) KDLブース展示用 アパレルEC ChatBot用 ファッションレコメンドAPIサーバ

![screen-shot](https://github.com/kobedigitallabo/078-fashion-recommend-api-server-demo/blob/master/screen-shot.png)

## API仕様

API仕様は [api/swagger/swagger.yaml](https://github.com/kobedigitallabo/078-fashion-recommend-api-server-demo/blob/master/api/swagger/swagger.yaml) に記載されています

## Setup

```
git clone https://github.com/bathtimefish/078-fashion-recommend-api-server-demo.git
cd 078-fashion-recommend-api-server-demo
npm install
```

APIサーバーは画像分類にPythonで作成された深層学習モデルを利用しています。Pythonおよび以下のモジュールをインストールしてください。

[tensorflow](https://github.com/tensorflow/tensorflow)  
[keras](https://github.com/fchollet/keras)  
[matplotlib](https://github.com/matplotlib/matplotlib)  
[pillow](https://github.com/python-pillow/Pillow)  
[numpy](https://github.com/numpy/numpy)  
[fire](https://github.com/google/python-fire)  

環境セットアップ時に上記モジュールをインストールする必要がある

セットアップが完了した後、`images`フォルダ下の`0`から`4`までのフォルダに服の画像データを5点以上配置します。
フォルダ名と服の種類の対応は以下のようになっています。各番号に対応する服画像を各フォルダ下に配置してください。

| フォルダ名   | 服種別     |
| :----------- | :-----     |
| 0            | コート     |
| 1            | スカート   |
| 2            | ワンピース |
| 3            | パンツ     |
| 4            | Tシャツ    |

## Run API Server

`swagger project start` or `node app.js`

## リクエスト方法

[example/test.html](https://github.com/kobedigitallabo/078-fashion-recommend-api-server-demo/blob/master/test.html) にリクエストのサンプルが記載されています。  

32x32pxの服画像(背景白色、正方形の画像で中央に服が一点掲載されている写真)をBASE64 encodeしたものを `http://[mydomain]/recommend` にPOSTします。  
APIサーバーはリクエストされた画像からラベルを推測し、ラベルに対応するフォルダ下の画像ファイル名をランダムに5つピックアップして以下の形式でレスポンスします。

```
{
  "data": {
    "label": 1,    // コート
    "images": [
      "10001.jpg",
      "10002.jpg",
      "10003.jpg",
      "10004.jpg",
      "10005.jpg",
    ]
  }
}
```

## Licence

Affero General Public License
