# financial-product-recommendation-system
This is a sample project for Financial Product Recommendation System


## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install python dependencies.

```bash
pip3 install -r requirements.txt
```

## Application Configuration
Before you begin, please update the configuration (./src/main/resources.application.yml)
```yaml
nn:
  file:
    #This is the File for processing
    input: /Users/frankngai/Desktop/workspace/financial-product-recommendation-system/src/main/resources/CUST_INVESTMENT.csv
    #This is the post process file
    output: /Users/frankngai/Desktop/workspace/financial-product-recommendation-system/output/data_reference.csv
  #This is the check point object of the neural network (weight only)
  checkpoint: /Users/frankngai/Desktop/workspace/financial-product-recommendation-system/output/
  #This is the complete model of the neural network
  model: /Users/frankngai/Desktop/workspace/financial-product-recommendation-system/output/recommendation_system_model.h5
```

## Usage
### Train the neural network
```bash
python3 ./src/main/python/rs-nn.py
```
### Startup API Server to server API Request
```bash
python3 ./src/main/python/rs-api.py
```

### User Recommendation API
![Image of User Recommendation API](api-doc/user_recommendation.png)
### Product Recommendation API
![Image of Product Recommendation API](api-doc/product_recommendation.png)
