from torch.utils.data import Dataset
from PIL import Image


# 이미지 전처리
class CustomImageDataset(Dataset):
    def __init__(self, file_list, label_list, img_T=None):
        """
            커스텀 이미지 데이터셋 클래스의 생성자입니다.

            Args:
                file_paths (list): 이미지 파일 경로의 리스트
                labels (list): 이미지 레이블의 리스트
                transform (callable, optional): 이미지에 적용할 전처리 함수
        """
        self.file_list = file_list
        self.label_list = label_list
        self.img_T = img_T

    def __getitem__(self, idx):
        """
            인덱스에 해당하는 샘플을 가져옵니다.

            Args:
                idx (int): 샘플의 인덱스

            Returns:
                image (torch.Tensor): 이미지 데이터의 텐서
                label (torch.Tensor): 이미지 레이블의 텐서
        """
        image = Image.open(self.file_list[idx])
        label = self.label_list[idx]

        if self.img_T is not None:
            image = self.img_T(image)

        return image, label

    def __len__(self):
        return len(self.file_list)
