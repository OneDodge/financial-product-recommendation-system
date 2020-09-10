# financial-product-recommendation-system
This is a sample project for Financial Product Recommendation System


## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install python dependencies.

```bash
pip3 install -r requirements.txt
```

## Dataset
If you want to get hold of the latest data set please visit [WealthML](https://github.com/twinzom/WealthML)

## Application Configuration
Before you begin, please update the configuration (./src/main/resources/application.yml)
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
### 1) Train the neural network
```bash
python3 ./src/main/python/rs-nn.py
```
### 2) Startup API Server to serve API Request
```bash
python3 ./src/main/python/rs-api.py
```

### User Recommendation API
#### End Point
```
http://127.0.0.1:5000/recommendation/user
```
#### Request Parameter
```
{
	"product_name": "U62300",
	"3year_return": "11.37",
	"standard_deviation": "20.12",
	"dividend": "0.22",
	"asset_class": "Equity Developed Market"
}
```
![Image of User Recommendation API](api-doc/user_recommendation.png)

### Product Recommendation API
#### End Point
```
http://127.0.0.1:5000/recommendation/product
```
#### Request Parameter
```
{
	"user": "CUST00000134",
	"age": 20,
	"gender": "M",
	"maritalStatus": "SINGLE",
	"haveChild": "N",
	"education": "SECONDARY"
}
```
![Image of Product Recommendation API](api-doc/product_recommendation.png)

### Data Visualisation Tool
![Image of Data Visualisation Tool](api-doc/visualisation.png)
