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
    input: /Users/xxx/Desktop/workspace/financial-product-recommendation-system/src/main/resources/CUST_INVESTMENT.csv
    output:
      #This is the check point object of the neural network (weight only)
      checkpoint: /Users/xxx/Desktop/workspace/financial-product-recommendation-system/output/checkpoint/
      #This is the complete model of the neural network
      model: /Users/xxx/Desktop/workspace/financial-product-recommendation-system/output/model/recommendation_system_model/
```
## Data Visualisation Configuration
Before you begin, please update the data visualisaton configuration (./visualisation.html) <br/>
If you are using default endpoint then you do not need to update this config
```
line 47: url = "http://localhost:5000/recommendation/data"
```

## Usage
### 1) Train the neural network
```bash
python3 ./src/main/python/rs_nn.py
```
### 2) Startup API Server to serve API Request
```bash
python3 ./src/main/python/rs_api.py
```

### User Recommendation API
#### End Point (Default)
```
http://127.0.0.1:5000/recommendation/user
```
#### Request Parameter
```
{
	"symbol": "2628.HK",
	"name": "China Life Insurance Company Limited",
	"price": "17.1",
	"change": "-0.1",
	"change_percentage": "-0.58",
	"market_captial": "1.2e+18",
	"trailing_p_e": "8.36",
	"revenue": "",
	"volume": "33552000.0",
	"total_cash": "",
	"total_debt": "",
	"5_year_average_dividend_yield": "1.96",
	"sector": "Financial Services",
	"industry": "Insurance—Life"
}
```
![Image of User Recommendation API](api-doc/user_recommendation.png)

### Product Recommendation API
#### End Point (Default)
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

### Production Architecture Diagram
![Image of Data Visualisation Tool](api-doc/architecture.png)
