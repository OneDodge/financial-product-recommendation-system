
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config


class UserService:
    @staticmethod
    def getUser(user):
        if Config.get()['userRepository']['live'] == True:
            return 1
        else:
            return Config.get()['userRepository']['fakeUser']


def main():
    print(UserService.getUser("1234"))


if __name__ == '__main__':
    main()
