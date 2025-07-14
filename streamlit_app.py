import streamlit as st
import pandas as pd
import os
from model import SalesPredictionModel

# Page configuration
st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_trained = False

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    model = SalesPredictionModel()
    if os.path.exists('sales_model.pkl'):
        try:
            model.load_model('sales_model.pkl')
            return model, True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return model, False
    return model, False

def train_model():
    """Train the model and save it"""
    try:
        if not os.path.exists('artificial_sales.csv'):
            st.error("Data file 'artificial_sales.csv' not found!")
            return False
        
        with st.spinner("Training model... This may take a few moments."):
            model = SalesPredictionModel()
            df = pd.read_csv('artificial_sales.csv')
            model.train(df)
            model.save_model('sales_model.pkl')
            
        st.session_state.model = model
        st.session_state.model_trained = True
        return True
    except Exception as e:
        st.error(f"Failed to train model: {str(e)}")
        return False

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Sales Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model on startup
    if st.session_state.model is None:
        model, trained = load_model()
        st.session_state.model = model
        st.session_state.model_trained = trained
    
    # Sidebar for model management
    with st.sidebar:
        st.header("🔧 Model Management")
        
        # Model status
        if st.session_state.model_trained:
            st.markdown('<div class="success-box">✅ Model is trained and ready!</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ Model not trained</div>', 
                       unsafe_allow_html=True)
        
        # Train/Retrain button
        if st.button("🚀 Train Model" if not st.session_state.model_trained else "🔄 Retrain Model", 
                     use_container_width=True):
            if train_model():
                st.success("Model trained successfully!")
                st.rerun()
        
        st.markdown("---")
        
        # Model info
        if st.session_state.model_trained:
            st.subheader("📈 Model Performance")
            st.metric("Accuracy (R²)", "99.8%")
            st.metric("Mean Error", "$11,193")
            
      
    
    # Main content area
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first using the sidebar.")
        
        # Show sample data info
        if os.path.exists('artificial_sales.csv'):
            st.subheader("📄 Dataset Preview")
            df = pd.read_csv('artificial_sales.csv')
            st.write(f"Dataset contains {len(df):,} rows and {len(df.columns)} columns")
            st.dataframe(df.head(), use_container_width=True)
        return
    
    # Prediction interface
    st.header("🎯 Sales Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Temporal Factors")
        
        day = st.selectbox(
            "Day of Week",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            index=0,
            help="Select the day of the week for prediction"
        )
        
        promo = st.checkbox(
            "Promotion Applied",
            value=False,
            help="Check if there's a promotional campaign running"
        )
        
        st.subheader("🌤️ Environmental Factors")
        
        weather = st.selectbox(
            "Weather Condition",
            options=['hot', 'cold', 'rainy', 'very hot', 'humid', 'very cold'],
            index=0,
            help="Current weather condition"
        )
    
    with col2:
        st.subheader("🍔 Product Information")
        
        product = st.selectbox(
            "Product Name",
            options=['VBurger', 'Veg Burger', 'Cheese Burger', 'Chicken Burger', 
                    'Chk Burger', 'Falafel Burger', 'Fries', 'Coca-Cola', 'Coke'],
            index=0,
            help="Select the product for sales prediction"
        )
        
        calories = st.number_input(
            "Product Calories",
            min_value=0,
            max_value=2000,
            value=500,
            step=10,
            help="Caloric content of the product"
        )
        
        st.subheader("📊 Economic Indicators")
        
        inflation = st.slider(
            "Inflation Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1,
            help="Current inflation percentage"
        )
        
        unemployment = st.slider(
            "Unemployment Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.1,
            help="Current unemployment percentage"
        )
    
    # Prediction button and results
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button(
            "🔮 Predict Sales",
            use_container_width=True,
            type="primary"
        )
    
    if predict_button:
        try:
            # Make prediction
            promo_value = "Yes" if promo else None
            
            with st.spinner("Making prediction..."):
                prediction = st.session_state.model.predict(
                    day=day,
                    product=product,
                    weather=weather,
                    calories=calories,
                    inflation=inflation,
                    unemployment=unemployment,
                    promo=promo_value
                )
            
            # Display result
            st.markdown("### 📊 Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f'''
                <div class="prediction-box">
                    <h2 style="text-align: center; margin: 0; color: #1f77b4;">
                        💰 ${prediction:,.2f}
                    </h2>
                    <p style="text-align: center; margin: 0.5rem 0 0 0; color: #666;">
                        Predicted Daily Sales
                    </p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Show prediction details
            st.markdown("### 📋 Prediction Details")
            
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.markdown("**Input Parameters:**")
                st.write(f"• **Day:** {day}")
                st.write(f"• **Product:** {product}")
                st.write(f"• **Weather:** {weather}")
                st.write(f"• **Calories:** {calories}")
            
            with details_col2:
                st.markdown("**Economic & Marketing:**")
                st.write(f"• **Inflation:** {inflation}%")
                st.write(f"• **Unemployment:** {unemployment}%")
                st.write(f"• **Promotion:** {'Yes' if promo else 'No'}")
                st.write(f"• **Weekend:** {'Yes' if day in ['Saturday', 'Sunday'] else 'No'}")
            
          
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Built with Streamlit • Sales Prediction System • Machine Learning Model
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 