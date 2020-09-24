
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config

PRODUCT_ASSET_CLASS_COLUMN = 'asset_class'
PRODUCT_ASSET_CLASS_INDEX_COLUMN = 'asset_class_index'

HEADERS = [PRODUCT_ASSET_CLASS_COLUMN, PRODUCT_ASSET_CLASS_INDEX_COLUMN]


class ProductAssetClassRepository:
    @staticmethod
    def generateIndex():
        names = Config.get()['productAssetClassRepository']['name'].split(",")

        df = pd.DataFrame(names)
        df.columns = [PRODUCT_ASSET_CLASS_COLUMN]

        df[PRODUCT_ASSET_CLASS_INDEX_COLUMN] = df[PRODUCT_ASSET_CLASS_COLUMN].astype(
            'category').cat.codes

        df.to_csv(Config.getNNIndex() +
                  "product_asset_class.index.csv", index=False)
        return df

    @staticmethod
    def readIndex():
        df = pd.read_csv(Config.getNNIndex() + "product_asset_class.index.csv",
                         sep=",",
                         names=HEADERS,
                         header=0,
                         dtype={
            PRODUCT_ASSET_CLASS_COLUMN: 'str',
            PRODUCT_ASSET_CLASS_INDEX_COLUMN: np.int32
        })
        return df


def main():
    print(ProductAssetClassRepository.generateIndex())
    print(ProductAssetClassRepository.readIndex())


if __name__ == '__main__':
    main()
