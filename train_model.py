# train_model.py
import pandas as pd
from model import SalesPredictionModel

def main():
    # Load data
    df = pd.read_csv('artificial_sales.csv')
    
    # Initialize and train model
    model = SalesPredictionModel()
    model.train(df)
    
    # Save trained model
    model.save_model('sales_model.pkl')
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    main()
