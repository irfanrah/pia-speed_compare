from io import BytesIO
from typing import Union

import boto3
import numpy as np
from PIL import Image


class DataLakeManager:
    """
    Data Lake 관리를 위한 클래스.

    오브젝트 스토리지(S3 호환)에 파일 및 이미지를 업로드하는 기능을 제공한다.
    """
    def __init__(
            self,
            access_key_id: str,
            secret_access_key: str,
            region_name: str,
            endpoint_url: str
    ):
        """
        DataLakeManager 생성자.

        Data Lake 세션을 생성하고, 클라이언트를 초기화한다.
        """
        self.session = boto3.session.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region_name,
        )
        self.client = self.session.client('s3', endpoint_url=endpoint_url)

    def upload_file(self, data, bucket_name, object_name, content_type='image/jpeg'):
        """
        파일을 Data Lake에 업로드.

        Args:
            data (bytes): 업로드할 파일 데이터.
            bucket_name (str): 버킷 이름.
            object_name (str): 저장될 객체 이름.
            content_type (str, optional): 파일의 Content-Type. 기본값은 'image/jpeg'.
        """
        self.client.put_object(
            Body=data, Bucket=bucket_name, Key=object_name, ContentType=content_type
        )

    def upload_image(self, image: np.array, bucket_name, object_name, content_type='image/jpeg'):
        """
        이미지(Numpy 배열)를 Data Lake에 업로드.

        Args:
            image (np.array): 업로드할 이미지 (OpenCV 형식, BGR 배열).
            bucket_name (str): 버킷 이름.
            object_name (str): 저장될 객체 이름.
            content_type (str, optional): 파일의 Content-Type. 기본값은 'image/jpeg'.
        """
        img_byte_arr = self.change_np2pil_jpg_bytes(image)
        self.upload_file(img_byte_arr, bucket_name, object_name, content_type=content_type)

    @staticmethod
    def change_np2pil_jpg_bytes(image, thumbnail_size:Union[None, str] = None):
        """
        Numpy 배열(BGR 형식)을 PIL 이미지로 변환하고 JPEG 바이트 형태로 반환.

        1. BGR -> RGB 변환
        2. 이미지 크기 조정
        3. JPEG 형식으로 메모리에 저장

        Args:
            image (np.array): 변환할 OpenCV 이미지 배열.
            thumbnail_size: 변환할 이미지 사이즈
                - str 또는 None만 가능, str = 248x124 와 같이 x로 나뉜 문자열

        Returns:
            bytes: JPEG 이미지 데이터.
        """

        # convert color (BGR -> RGB)
        image = image[..., ::-1]
        # resize image
        image = Image.fromarray(image)
        if thumbnail_size is not None:
            image = image.resize(list(map(int, thumbnail_size.split("x"))))
        # 메모리에 이미지 저장
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')  # JPEG 형식으로 저장
        return img_byte_arr.getvalue()
