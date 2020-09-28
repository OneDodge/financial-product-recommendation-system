import yaml
import os
from pathlib import Path

CONFIG_PATH = os.path.abspath("%s%s" % (
    Path(os.path.join(os.path.dirname(__file__))).parent, "/resources/application.yml"))

with open(CONFIG_PATH) as file:
    cf = yaml.load(file, Loader=yaml.FullLoader)


class Config:
    @staticmethod
    def getNNProductFileInput():
        return cf['nn']['product']['file']['input']

    @staticmethod
    def getNNPreProcessingCustomerFileInput():
        return cf['nn']['pre-processing']['customer']['file']['input']

    @staticmethod
    def getNNPreProcessingProductFileInput():
        return cf['nn']['pre-processing']['product']['file']['input']

    @staticmethod
    def getNNPreProcessingFileOutput():
        return cf['nn']['pre-processing']['file']['output']

    @staticmethod
    def getNNProcessingFileInput():
        return cf['nn']['processing']['file']['input']

    @staticmethod
    def getNNCheckpoint():
        return cf['nn']['file']['output']['checkpoint']

    @staticmethod
    def getNNModel():
        return cf['nn']['file']['output']['model']

    @staticmethod
    def get():
        return cf


def main():
    print(cf)


if __name__ == '__main__':
    main()
