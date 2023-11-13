# torchvision.transform 메소드 정리

* Resize: 이미지의 크기를 조절합니다.
* RandomResizedCrop: 이미지를 무작위로 자르고 크기를 조절합니다.
* RandomHorizontalFlip: 이미지를 무작위로 수평으로 뒤집습니다.
* RandomVerticalFlip: 이미지를 무작위로 수직으로 뒤집습니다.
* ToTensor: 이미지를 텐서로 변환합니다.
* Normalize: 이미지를 정규화합니다.
* ColorJitter: 이미지의 색상을 무작위로 조정합니다.
* RandomRotation: 이미지를 무작위로 회전합니다.
* RandomCrop: 이미지를 무작위로 자릅니다.
* Grayscale: 이미지를 흑백으로 변환합니다.
* RandomSizedCrop: 이미지를 무작위로 자르고 크기를 조절합니다.

### 예시

```
    # 이미지 전처리 작업을 정의
    preprocess = T.Compose([
        # 256x256 으로 이미지 사이즈 조정
        T.Resize((256, 256)),
        # 무작위 수평 뒤집기
        T.RandomHorizontalFlip(),
        # 텐서로 변환
        T.ToTensor(),
        # 정규화
        T.Normalize((0.5), (0.5))# T.Normalize([0.5],[0.5])
    ])
```