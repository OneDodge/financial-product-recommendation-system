
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config
from productAssetClassRepository import ProductAssetClassRepository
import productAssetClassRepository

df = ProductAssetClassRepository.readIndex()


class ProductAssetClassService:
    @staticmethod
    def getProductAssetClass(productAssetClass):
        if df[df[productAssetClassRepository.PRODUCT_ASSET_CLASS_COLUMN] == productAssetClass].empty:
            return df[productAssetClassRepository.PRODUCT_ASSET_CLASS_INDEX_COLUMN].max() + 1
        else:
            return df[df[productAssetClassRepository.PRODUCT_ASSET_CLASS_COLUMN] == productAssetClass][productAssetClassRepository.PRODUCT_ASSET_CLASS_INDEX_COLUMN].to_numpy()[0]


def main():
    print(ProductAssetClassService.getProductAssetClass('Equity Developed Market'))
    # print(ProductAssetClassService.getProductAssetClass('MARRIED'))


if __name__ == '__main__':
    main()
