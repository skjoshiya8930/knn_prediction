# app.py

import os
import joblib
import traceback
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
try:
    deployed_knn = joblib.load('obesity_knn_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model not found or error loading. {e}")
    deployed_knn = None

# ==========================================================
# Prediction Function with Bulletproof Error Handling
# ==========================================================
def predict_obesity(
    Age, Gender, Height, Weight, CALC, SCC, CAEC, FAVC, FCVC, NCP,
    SMOKE, CH2O, family_history_with_overweight, FAF, TUE, MTRANS
):
    values = [
        Age, Gender, Height, Weight, CALC, SCC, CAEC, FAVC, FCVC, NCP,
        SMOKE, CH2O, family_history_with_overweight, FAF, TUE, MTRANS
    ]

    # 1. Empty input check 
    if any(v is None or str(v).strip() == "" for v in values):
        return "Please fill in all the input fields and select all dropdown options."

    # 2. Type casting
    try:
        Age = float(Age)
        Gender = int(Gender) 
        Height = float(Height)
        Weight = float(Weight)
        CALC = int(CALC)
        SCC = int(SCC)
        CAEC = int(CAEC)
        FAVC = int(FAVC)
        FCVC = float(FCVC)
        NCP = float(NCP)
        SMOKE = int(SMOKE)
        CH2O = float(CH2O)
        family_history_with_overweight = int(family_history_with_overweight)
        FAF = float(FAF)
        TUE = float(TUE)
        MTRANS = int(MTRANS)
    except (ValueError, TypeError) as e:
        return f"❌ Data Conversion Error. Expected numbers.\n\nDetails: {str(e)}"

    # 3. Negative value check
    if any(v < 0 for v in [Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE]):
        return "❌ Negative values are not allowed for physical metrics."

    # 4. Model execution
    if deployed_knn is None:
        return "❌ Model failed to load. Please check your .pkl file and requirements."

    try:
        input_data = [[
            Age, Gender, Height, Weight, CALC, SCC, CAEC, FAVC, FCVC, NCP,
            SMOKE, CH2O, family_history_with_overweight, FAF, TUE, MTRANS
        ]]

        prediction = deployed_knn.predict(input_data)
        
        result_class = str(prediction[0]).replace("_", " ")
        return f"🩺 Assessment Result\n\nPredicted Category: {result_class}"

    except Exception as e:
        error_trace = traceback.format_exc()
        print("RUNTIME ERROR:\n", error_trace)
        return f"❌ Prediction failed due to an internal error.\n\nDEBUG INFO:\n{error_trace}"

# ==========================================================
# Interface Setup (Enhanced Layout)
# ==========================================================
# --- CODE BLOCK: ENHANCED UI WITH GR.BLOCKS ---
# Using gr.Blocks for custom horizontal layout and theming
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as app:
    
    # Header
    gr.Markdown("<h1 style='text-align: center;'>🩺 Obesity Level Estimation System</h1>")
    gr.Markdown("<p style='text-align: center; font-size: 16px;'>Estimate obesity levels based on eating habits and physical condition using a trained <b>K-Nearest Neighbors (KNN)</b> model.</p>")
    gr.Markdown("---")

    # Layout: Three horizontal columns inside one main row
    with gr.Row():
        
        # Column 1: Physical Attributes
        with gr.Column():
            gr.Markdown("### 👤 Personal & Physical Attributes")
            Age = gr.Number(label="Age (Years)")
            Gender = gr.Dropdown(choices=[("Female", 0), ("Male", 1)], label="Gender")
            Height = gr.Number(label="Height (Meters, e.g., 1.75)")
            Weight = gr.Number(label="Weight (Kilograms)")
            family_history_with_overweight = gr.Dropdown(choices=[("No", 0), ("Yes", 1)], label="Family History with Overweight")

        # Column 2: Dietary Habits
        with gr.Column():
            gr.Markdown("### 🥗 Dietary Habits")
            FAVC = gr.Dropdown(choices=[("No", 0), ("Yes", 1)], label="High Caloric Food (FAVC)")
            FCVC = gr.Number(label="Vegetables in Meals (FCVC, 1 to 3)")
            NCP = gr.Number(label="Main Meals Daily (NCP, 1 to 4)")
            CAEC = gr.Dropdown(choices=[("No", 0), ("Sometimes", 1), ("Frequently", 2), ("Always", 3)], label="Food Between Meals (CAEC)")
            CALC = gr.Dropdown(choices=[("No", 0), ("Sometimes", 1), ("Frequently", 2), ("Always", 3)], label="Alcohol Consumption (CALC)")
            CH2O = gr.Number(label="Water Daily (CH2O, Liters)")

        # Column 3: Lifestyle
        with gr.Column():
            gr.Markdown("### 🏃‍♂️ Lifestyle & Activity")
            SMOKE = gr.Dropdown(choices=[("No", 0), ("Yes", 1)], label="Smoker (SMOKE)")
            FAF = gr.Number(label="Physical Activity Frequency (FAF, 0 to 3)")
            TUE = gr.Number(label="Technology Use Time (TUE, 0 to 2)")
            SCC = gr.Dropdown(choices=[("No", 0), ("Yes", 1)], label="Calorie Monitoring (SCC)")
            MTRANS = gr.Dropdown(choices=[("Automobile", 0), ("Motorbike", 1), ("Bike", 2), ("Public Transit", 3), ("Walking", 4)], label="Transportation (MTRANS)")

    # Output Section
    gr.Markdown("---")
    with gr.Row():
        submit_btn = gr.Button("Evaluate Obesity Level", variant="primary", size="lg")
        clear_btn = gr.ClearButton(size="lg")
    
    with gr.Row():
        result_box = gr.Textbox(label="Assessment Result", lines=3, interactive=False)

    # Footer
    gr.Markdown("""
    ---
    ### 👨‍💻 About the Developer
    **Created by:** Sachin (MERN Stack Developer & SDE)
    * **LinkedIn:** [Connect with me](YOUR_LINKEDIN_URL_HERE)
    * **GitHub:** [Check out my projects](YOUR_GITHUB_URL_HERE)
    """)

    # --- Wire up the logic ---
    # Ensure this array exactly matches the 16 arguments of predict_obesity!
    input_components = [
        Age, Gender, Height, Weight, CALC, SCC, CAEC, FAVC, FCVC, NCP,
        SMOKE, CH2O, family_history_with_overweight, FAF, TUE, MTRANS
    ]
    
    submit_btn.click(fn=predict_obesity, inputs=input_components, outputs=result_box)
    clear_btn.add(input_components + [result_box])
# ----------------------------------------------

# ==========================================================
# Launch Configuration
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Gradio server on 0.0.0.0:{port}...")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
