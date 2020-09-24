
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config


class ProductService:
    @staticmethod
    def getProduct(product):
        if Config.get()['productRepository']['live'] == True:
            return 1
        else:
            return Config.get()['productRepository']['fakeProduct']


def main():
    print(ProductService.getProduct("1234"))


if __name__ == '__main__':
    main()
