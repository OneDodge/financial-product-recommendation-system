import yaml
import os
from pathlib import Path

CONFIG_PATH = os.path.abspath("%s%s" % (
    Path(os.path.join(os.path.dirname(__file__))).parent, "/resources/application.yml"))

with open(CONFIG_PATH) as file:
    cf = yaml.load(file, Loader=yaml.FullLoader)


class Config:
    @staticmethod
    def getNNFileInput():
        return cf['nn']['file']['input']

    @staticmethod
    def getNNFileOutput():
        return cf['nn']['file']['output']

    @staticmethod
    def getNNCheckpoint():
        return cf['nn']['checkpoint']

    @staticmethod
    def getNNModel():
        return cf['nn']['model']


def main():
    print(cf)


if __name__ == '__main__':
    main()
