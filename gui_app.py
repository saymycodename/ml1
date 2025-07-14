import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from model import SalesPredictionModel
import os

class SalesPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Prediction System")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize model
        self.model = SalesPredictionModel()
        self.model_trained = False
        
        # Create GUI
        self.create_widgets()
        
        # Check if model exists and load it
        self.load_model_if_exists()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Sales Prediction System", 
                              font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Model training section
        train_frame = tk.LabelFrame(main_frame, text="Model Training", 
                                   font=("Arial", 12, "bold"), bg='#f0f0f0')
        train_frame.pack(fill='x', pady=10)
        
        self.train_button = tk.Button(train_frame, text="Train Model", 
                                     command=self.train_model, 
                                     bg='#4CAF50', fg='white', 
                                     font=("Arial", 10, "bold"))
        self.train_button.pack(pady=10)
        
        self.train_status = tk.Label(train_frame, text="Model not trained", 
                                    bg='#f0f0f0', fg='red')
        self.train_status.pack()
        
        # Prediction section
        pred_frame = tk.LabelFrame(main_frame, text="Sales Prediction", 
                                  font=("Arial", 12, "bold"), bg='#f0f0f0')
        pred_frame.pack(fill='both', expand=True, pady=10)
        
        # Input fields
        fields_frame = tk.Frame(pred_frame, bg='#f0f0f0')
        fields_frame.pack(pady=10)
        
        # Day
        tk.Label(fields_frame, text="Day:", bg='#f0f0f0').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.day_var = tk.StringVar()
        day_combo = ttk.Combobox(fields_frame, textvariable=self.day_var, 
                                values=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                       'Friday', 'Saturday', 'Sunday'])
        day_combo.grid(row=0, column=1, padx=5, pady=5)
        day_combo.set('Monday')
        
        # Product
        tk.Label(fields_frame, text="Product:", bg='#f0f0f0').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.product_var = tk.StringVar()
        product_combo = ttk.Combobox(fields_frame, textvariable=self.product_var,
                                    values=['VBurger', 'Veg Burger', 'Cheese Burger', 'Chicken Burger', 
                                           'Chk Burger', 'Falafel Burger', 'Fries', 'Coca-Cola', 'Coke'])
        product_combo.grid(row=1, column=1, padx=5, pady=5)
        product_combo.set('VBurger')
        
        # Weather
        tk.Label(fields_frame, text="Weather:", bg='#f0f0f0').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.weather_var = tk.StringVar()
        weather_combo = ttk.Combobox(fields_frame, textvariable=self.weather_var,
                                    values=['hot', 'cold', 'rainy', 'very hot', 'humid', 'very cold'])
        weather_combo.grid(row=2, column=1, padx=5, pady=5)
        weather_combo.set('hot')
        
        # Calories
        tk.Label(fields_frame, text="Calories:", bg='#f0f0f0').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.calories_var = tk.StringVar(value="500")
        calories_entry = tk.Entry(fields_frame, textvariable=self.calories_var)
        calories_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Inflation
        tk.Label(fields_frame, text="Inflation %:", bg='#f0f0f0').grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.inflation_var = tk.StringVar(value="3.5")
        inflation_entry = tk.Entry(fields_frame, textvariable=self.inflation_var)
        inflation_entry.grid(row=4, column=1, padx=5, pady=5)
        
        # Unemployment
        tk.Label(fields_frame, text="Unemployment %:", bg='#f0f0f0').grid(row=5, column=0, sticky='w', padx=5, pady=5)
        self.unemployment_var = tk.StringVar(value="5.0")
        unemployment_entry = tk.Entry(fields_frame, textvariable=self.unemployment_var)
        unemployment_entry.grid(row=5, column=1, padx=5, pady=5)
        
        # Promotion
        tk.Label(fields_frame, text="Promotion:", bg='#f0f0f0').grid(row=6, column=0, sticky='w', padx=5, pady=5)
        self.promo_var = tk.BooleanVar()
        promo_check = tk.Checkbutton(fields_frame, variable=self.promo_var, bg='#f0f0f0')
        promo_check.grid(row=6, column=1, padx=5, pady=5, sticky='w')
        
        # Predict button
        predict_button = tk.Button(pred_frame, text="Predict Sales", 
                                  command=self.predict_sales,
                                  bg='#2196F3', fg='white', 
                                  font=("Arial", 12, "bold"))
        predict_button.pack(pady=20)
        
        # Result
        self.result_label = tk.Label(pred_frame, text="", 
                                    font=("Arial", 14, "bold"), 
                                    bg='#f0f0f0', fg='#333')
        self.result_label.pack(pady=10)
    
    def load_model_if_exists(self):
        if os.path.exists('sales_model.pkl'):
            try:
                self.model.load_model('sales_model.pkl')
                self.model_trained = True
                self.train_status.config(text="Model loaded successfully", fg='green')
                self.train_button.config(text="Retrain Model")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def train_model(self):
        try:
            # Check if data file exists
            if not os.path.exists('artificial_sales.csv'):
                messagebox.showerror("Error", "Data file 'artificial_sales.csv' not found!")
                return
            
            # Show training message
            self.train_status.config(text="Training model...", fg='orange')
            self.root.update()
            
            # Load data and train model
            df = pd.read_csv('artificial_sales.csv')
            self.model.train(df)
            
            # Save model
            self.model.save_model('sales_model.pkl')
            self.model_trained = True
            
            self.train_status.config(text="Model trained successfully!", fg='green')
            self.train_button.config(text="Retrain Model")
            
            messagebox.showinfo("Success", "Model trained and saved successfully!")
            
        except Exception as e:
            self.train_status.config(text="Training failed", fg='red')
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def predict_sales(self):
        if not self.model_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
        
        try:
            # Get input values
            day = self.day_var.get()
            product = self.product_var.get()
            weather = self.weather_var.get()
            calories = float(self.calories_var.get())
            inflation = float(self.inflation_var.get())
            unemployment = float(self.unemployment_var.get())
            promo = "Yes" if self.promo_var.get() else None
            
            # Make prediction
            prediction = self.model.predict(day, product, weather, calories, 
                                          inflation, unemployment, promo)
            
            # Display result
            self.result_label.config(text=f"Predicted Sales: ${prediction:.2f}")
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values for calories, inflation, and unemployment!")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

def main():
    root = tk.Tk()
    app = SalesPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
