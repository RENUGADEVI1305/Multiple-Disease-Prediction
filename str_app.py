import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Multiple Disease Prediction")

st.sidebar.title("My Dashboard")
page=st.sidebar.radio('Visit',["Home", "Project Explanation", "Predicting Parkinsons Disease", "Predicting Kidney Disease", "Predicting Liver Disease",  "Developer Info"])

#-------------------------------------------------------------------------------------------------------------------------
   # page1 -  Home 

if page == "Home":
    
    st.header("Home")
    st.markdown(" ## Mini project 4:  ")
    st.markdown(" ### Title - Multiple Disease Prediction")
    st.image("D:/GUVI/project4/PIC1.jpeg", width=600)

#-------------------------------------------------------------------------------------------------------------------------------
  # page2 -  Project Explanation 

elif page == "Project Explanation":
    st.header("Project Explanation")
    #st.image("D:/GUVI/project1/images/pic2.jpg", width=300)
    st.write(""" **Objective:**  To build a scalable and accurate system that:
                     
             ● Assists in early detection of diseases.
                     
             ● Improves decision-making for healthcare providers.
                     
             ● Reduces diagnostic time and cost by providing quick predictions. """)
    
    st.write(""" **AIM:** Multi-disease Prediction: Predicts the likelihood of multiple diseases (e.g., Parkinsons, Kidney Disease, Liver Disease).
            
              1. Predict the Parkinsons Disease.
             
              2. Predict the Kidney Disease.
             
              3. Predict the Indian Liver Disease.
             
             """)
    
#---------------------------------------------------------------------------------------------------------------
    # page3 -  Predicting Parkinsons Disease 

elif page == "Predicting Parkinsons Disease":
    st.header("🧠 Predicting Parkinsons Disease")
    st.write("Provide patient’s voice and biomedical signal measurements below:")
 


    MODEL_PATH = "D:/GUVI/project4/myenv/Scripts/parkinsons_pipeline.joblib"

    # ---- Load model and metadata ----
    if os.path.exists(MODEL_PATH):
        try:
            saved = joblib.load(MODEL_PATH)
            pipe = saved.get('pipeline')  # imblearn pipeline
            numeric_features = saved.get('numeric_features', [])
            categorical_features = saved.get('categorical_features', [])
            feature_names = saved.get('feature_names', [])
            st.success(f"✅ Loaded pipeline from {MODEL_PATH}")
        except Exception as e:
            st.error(f"Failed to load pipeline file: {e}")
            st.stop()
    else:
        st.warning(f"No model file found at '{MODEL_PATH}'. Please train & save a model first.")
        st.stop()

    # Extract preprocessor and model (SMOTE is skipped during predict)
    try:
        preprocessor = pipe.named_steps['preprocess']
        classifier = pipe.named_steps['model']
    except Exception:
        st.error("Pipeline missing expected steps ('preprocess' and 'model'). Please confirm training pipeline step names.")
        st.stop()

    st.write("Model ready. You can enter a single sample manually or upload a CSV for batch prediction.")

    # ---- Helper function for preparing inputs----
    def prepare_input(df_input: pd.DataFrame) -> np.ndarray:
        """
        Preprocesses new input data before prediction.
        """
        df = df_input.copy()

        # Check and fill missing expected columns
        expected_cols =feature_names
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}. Uploaded data must contain these columns: {expected_cols}")

        # Keep only the columns used by preprocessor, in any order (ColumnTransformer selects by name)
        # Preprocessor expects numeric_features and categorical ones (excl OverTime if you used label encoding)
        # So pass the DataFrame as-is (preprocessor will internally pick columns by name)
        X_prepared = preprocessor.transform(df)  # returns numpy array
        return X_prepared

    # ---- Option A: Manual single-row input ----
    st.subheader("Manual Input (Single Patient)")
    with st.form(key='manual_input_form'):
        cols = st.columns(4)
        
        spread1 = cols[0].number_input("spread1", -7.5, -3.0, -5.7, step=0.1)
        PPE = cols[0].number_input("PPE", 0.09, 0.37, 0.20, step=0.01)
        spread2 = cols[0].number_input("spread2", 0.08, 0.37, 0.22, step=0.01)

        MDVP_Shimmer = cols[1].number_input("MDVP:Shimmer", 0.01, 0.07, 0.03, step=0.005)
        MDVP_APQ = cols[1].number_input("MDVP:APQ", 0.01, 0.06, 0.02, step=0.005)
        MDVP_Jitter_Abs = cols[1].number_input("MDVP:Jitter(Abs)", 0.00001, 0.0001, 0.00004, step=0.00001)

        Shimmer_APQ5 = cols[2].number_input("Shimmer:APQ5", 0.006, 0.043, 0.017, step=0.001)
        MDVP_Shimmer_dB = cols[2].number_input("MDVP:Shimmer(dB)", 0.10, 0.65, 0.27, step=0.01)
        Shimmer_APQ3 = cols[2].number_input("Shimmer:APQ3", 0.005, 0.036, 0.015, step=0.001)
        
        Shimmer_DDA = cols[3].number_input("Shimmer:DDA", 0.016, 0.109, 0.045, step=0.005)
        MDVP_PPQ = cols[3].number_input("MDVP:PPQ", 0.001, 0.009, 0.003, step=0.001)
        D2 = cols[3].number_input("D2", 1.8, 3.1, 2.38, step=0.05)
        
        submit_manual = st.form_submit_button("🔍 Predict (manual)")

    if submit_manual:
        try:
            input_df = pd.DataFrame([{
                'spread1': spread1,
                'PPE': PPE,
                'spread2': spread2,
                'MDVP:Shimmer': MDVP_Shimmer,
                'MDVP:APQ': MDVP_APQ,
                'MDVP:Jitter(Abs)': MDVP_Jitter_Abs,
                'Shimmer:APQ5': Shimmer_APQ5,
                'MDVP:Shimmer(dB)': MDVP_Shimmer_dB,
                'Shimmer:APQ3': Shimmer_APQ3,
                'Shimmer:DDA': Shimmer_DDA,
                'MDVP:PPQ': MDVP_PPQ,
                'D2': D2
            }])

            X_input = prepare_input(input_df)
            pred = classifier.predict(X_input)
            pred_proba = classifier.predict_proba(X_input)[:, 1] if hasattr(classifier, "predict_proba") else None

           

            # 💡 Stylish output card
            
            st.subheader("🎯 Prediction Result")

            if pred[0] == 1:
               st.markdown(
                f"""
                <div style='background-color:#ffe6e6;padding:20px;border-radius:12px;text-align:center'>
                        <h3 style='color:#cc0000;'>⚠️ Parkinson's Detected</h3>
                        <p style='font-size:18px;'>This patient is <b>likely to have Parkinson’s disease</b>.</p>
                        <p><b>Probability:</b> {pred_proba[0]:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
               st.markdown(
                f"""
                <div style='background-color:#e6ffe6;padding:20px;border-radius:12px;text-align:center'>
                        <h3 style='color:#007700;'>✅ No Parkinson's Detected</h3>
                        <p style='font-size:18px;'>This patient is <b>unlikely to have Parkinson’s disease</b>.</p>
                        <p><b>Probability:</b> {pred_proba[0]:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ---- Option B: Batch CSV upload ----
    
    st.subheader("Batch prediction (CSV Upload)")
    batch_file = st.file_uploader("Upload CSV containing feature columns", type=['csv'], key='batch_upload_parkinsons')

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())

            if st.button("Run batch prediction"):
                try:
                    X_batch = prepare_input(batch_df)
                    preds = classifier.predict(X_batch)
                    probs = classifier.predict_proba(X_batch)[:, 1] if hasattr(classifier, "predict_proba") else None

                    # attach results
                    batch_results = batch_df.copy()
                    batch_results['Predicted_Status'] = ['Parkinsons' if p==1 else 'Healthy' for p in preds]
                    if probs is not None:
                        batch_results['Probability'] = probs

                    st.success("✅ Batch prediction completed!")
                    st.dataframe(batch_results)

                    # Provide download link
                    csv = batch_results.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download results CSV", data=csv, file_name='parkinsons_predictions.csv', mime='text/csv')

                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

    st.markdown("---")
    st.caption("🧩 **Note:** Ensure uploaded data has the same feature columns used during model training.")

#---------------------------------------------------------------------------------------------------------------------------

   # page4 -  Predicting Kidney Disease 

elif page == "Predicting Kidney Disease":
    st.header("🧬 Predicting Kidney Disease")
    st.write("Upload patient data or enter details manually to predict CKD (Chronic Kidney Disease) presence (1 = CKD, 0 = Not CKD).")
    
  # ----------------------------
  #  Load Model
  # ----------------------------
    MODEL_PATH = "D:/GUVI/project4/myenv/Scripts/kidney_pipeline.joblib"

    
    if os.path.exists(MODEL_PATH):
       try:
        saved = joblib.load(MODEL_PATH)
        pipe = saved.get('pipeline')  # imblearn pipeline
        numeric_features = saved.get('numeric_features', [])
        categorical_features = saved.get('categorical_features', [])
        feature_names = saved.get('feature_names', [])
        st.success(f"✅ Loaded pipeline from {MODEL_PATH}")
       except Exception as e:
        st.error(f"Failed to load pipeline file: {e}")
        st.stop()
    else:
       st.warning(f"No model file found at '{MODEL_PATH}'. Please train & save a model first.")
       st.stop()

      # Extract preprocessor and classifier
    try:
       preprocessor = pipe.named_steps['preprocess']
       classifier = pipe.named_steps['model']
    except Exception:
       st.error("Pipeline missing expected steps ('preprocess' and 'model'). Please confirm training pipeline step names.")
       st.stop()

    st.write("Model ready. You can enter a single patient manually or upload a CSV for batch prediction.")
    
    
            # ----------------------------
              # Helper function
              # ----------------------------
    def prepare_input(df_input: pd.DataFrame) -> np.ndarray:
      """
      Preprocess new input data before prediction.
      """
      df = df_input.copy()
      expected_cols = feature_names
      missing = [c for c in expected_cols if c not in df.columns]
      if missing:
        raise ValueError(f"Missing expected columns: {missing}. Uploaded data must contain these columns: {expected_cols}")
      X_prepared = preprocessor.transform(df)
      return X_prepared


         # ----------------------------
          # Option A: Manual Input
           # ----------------------------
    st.subheader("Manual Input (Single Patient)")
    with st.form(key='manual_input_form_kidney'):
      cols = st.columns(3)

      # Numeric Inputs
      sc = cols[0].number_input("Serum Creatinine (sc)", min_value=0.0, step=0.1, value=1.2)
      al = cols[0].number_input("Albumin (al)", min_value=0.0, step=0.1, value=1.0)
      bu = cols[0].number_input("Blood Urea (bu)", min_value=0.0, step=1.0, value=45.0)
      bgr = cols[1].number_input("Blood Glucose Random (bgr)", min_value=0.0, step=1.0, value=120.0)
      bp = cols[1].number_input("Blood Pressure (bp)", min_value=0.0, step=1.0, value=80.0)
      age = cols[1].number_input("Age", min_value=0, step=1, value=50)
      su = cols[2].number_input("Sugar (su)", min_value=0.0, step=0.1, value=0.0)
      wc = cols[2].number_input("White Blood Cell Count (wc)", min_value=0.0, step=100.0, value=8000.0)

      # Categorical Inputs
      htn = cols[0].selectbox("Hypertension (htn)", ['yes', 'no'])
      dm = cols[1].selectbox("Diabetes Mellitus (dm)", ['yes', 'no'])
      appet = cols[2].selectbox("Appetite (appet)", ['good', 'poor'])
      pc = cols[2].selectbox("Pus Cell (pc)", ['normal', 'abnormal'])

      submit_manual = st.form_submit_button("🔍 Predict (manual)")

    if submit_manual:
      try:
        input_df = pd.DataFrame([{
            'sc': sc,
            'al': al,
            'bu': bu,
            'bgr': bgr,
            'bp': bp,
            'age': age,
            'su': su,
            'wc': wc,
            'htn': htn,
            'dm': dm,
            'appet': appet,
            'pc': pc
        }])

        X_input = prepare_input(input_df)
        pred = classifier.predict(X_input)
        pred_proba = classifier.predict_proba(X_input)[:, 1] if hasattr(classifier, "predict_proba") else None

        # 🎯 Stylish result display
        st.subheader("🎯 Prediction Result")

        if pred[0] == 1:
            st.markdown(
                f"""
                <div style='background-color:#ffe6e6;padding:20px;border-radius:12px;text-align:center'>
                    <h3 style='color:#cc0000;'>⚠️ CKD Detected</h3>
                    <p style='font-size:18px;'>This patient is <b>likely to have Chronic Kidney Disease</b>.</p>
                    <p><b>Probability:</b> {pred_proba[0]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style='background-color:#e6ffe6;padding:20px;border-radius:12px;text-align:center'>
                    <h3 style='color:#007700;'>✅ No CKD Detected</h3>
                    <p style='font-size:18px;'>This patient is <b>unlikely to have Chronic Kidney Disease</b>.</p>
                    <p><b>Probability:</b> {pred_proba[0]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

      except Exception as e:
        st.error(f"Prediction failed: {e}")


            # ----------------------------
            # Option B: Batch Prediction
            # ----------------------------
    st.subheader("📂 Batch Prediction (CSV Upload)")
    batch_file = st.file_uploader("Upload a CSV containing patient data", type=['csv'], key='batch_upload_kidney')

    if batch_file is not None:
      try:
        batch_df = pd.read_csv(batch_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        if st.button("Run batch prediction"):
            try:
                X_batch = prepare_input(batch_df)
                preds = classifier.predict(X_batch)
                probs = classifier.predict_proba(X_batch)[:, 1] if hasattr(classifier, "predict_proba") else None

                # Add results
                batch_results = batch_df.copy()
                batch_results['Predicted_Status'] = ['CKD' if p == 1 else 'No CKD' for p in preds]
                if probs is not None:
                    batch_results['Probability'] = probs

                st.success("✅ Batch prediction completed successfully!")
                st.dataframe(batch_results.head())

                csv = batch_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name='kidney_predictions.csv',
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
      except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

    st.markdown("---")
    st.caption("🧩 **Note:** Ensure the uploaded CSV has the same feature columns used during training.")


#------------------------------------------------------------------------------------------------------------------------------

  # page5 -  Predicting Liver Disease 

elif page == "Predicting Liver Disease":
    st.header("🧬 Predicting Indian Liver Disease") 
    st.write("Upload patient data or enter details manually to predict **Liver Disease** presence (1 = Patient, 2 = Not Patient).")

     # ----------------------------
     #  Load Model
     # ----------------------------
    MODEL_PATH = "D:/GUVI/project4/myenv/Scripts/liver_pipeline.joblib"

    if os.path.exists(MODEL_PATH):
      try:
        saved = joblib.load(MODEL_PATH)
        pipe = saved.get('pipeline')
        numeric_features = saved.get('numeric_features', [])
        categorical_features = saved.get('categorical_features', [])
        feature_names = saved.get('feature_names', [])
        st.success(f"✅ Loaded pipeline from {MODEL_PATH}")
      except Exception as e:
        st.error(f"Failed to load pipeline file: {e}")
        st.stop()
    else:
      st.warning(f"No model file found at '{MODEL_PATH}'. Please train & save a model first.")
      st.stop()

           # Extract preprocessor and classifier
    try:
      preprocessor = pipe.named_steps['preprocess']
      classifier = pipe.named_steps['model']
    except Exception:
      st.error("Pipeline missing expected steps ('preprocess' and 'model'). Please confirm training pipeline step names.")
      st.stop()

    st.write("Model ready. You can enter a single patient manually or upload a CSV for batch prediction.")

             # ----------------------------
            # Helper function
            # ----------------------------
    def prepare_input(df_input: pd.DataFrame) -> np.ndarray:
      """
      Preprocess new input data before prediction.
      """
      df = df_input.copy()
      expected_cols = feature_names
      missing = [c for c in expected_cols if c not in df.columns]
      if missing:
        raise ValueError(f"Missing expected columns: {missing}. Uploaded data must contain these columns: {expected_cols}")
      X_prepared = preprocessor.transform(df)
      return X_prepared

         # ----------------------------
          # Option A: Manual Input
         # ----------------------------
    st.subheader("Manual Input (Single Patient)")

    with st.form(key='manual_input_form_liver'):
      cols = st.columns(3)

      # Numeric Inputs
      age = cols[0].number_input("Age", min_value=1, max_value=120, step=1, value=45)
      total_bilirubin = cols[0].number_input("Total Bilirubin", min_value=0.0, step=0.1, value=1.0)
      direct_bilirubin = cols[0].number_input("Direct Bilirubin", min_value=0.0, step=0.1, value=0.3)
      alkaline_phosphotase = cols[1].number_input("Alkaline Phosphotase", min_value=0.0, step=1.0, value=200.0)
      alamine_aminotransferase = cols[1].number_input("Alamine Aminotransferase", min_value=0.0, step=1.0, value=30.0)
      aspartate_aminotransferase = cols[1].number_input("Aspartate Aminotransferase", min_value=0.0, step=1.0, value=40.0)
      total_protiens = cols[2].number_input("Total Proteins", min_value=0.0, step=0.1, value=6.8)
      albumin = cols[2].number_input("Albumin", min_value=0.0, step=0.1, value=3.5)
      albumin_globulin_ratio = cols[2].number_input("Albumin and Globulin Ratio", min_value=0.0, step=0.1, value=1.0)

      # Categorical Input
      gender = cols[0].selectbox("Gender", ['Male', 'Female'])

      submit_manual = st.form_submit_button("🔍 Predict (manual)")

    if submit_manual:
      try:
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Total_Bilirubin': total_bilirubin,
            'Direct_Bilirubin': direct_bilirubin,
            'Alkaline_Phosphotase': alkaline_phosphotase,
            'Alamine_Aminotransferase': alamine_aminotransferase,
            'Aspartate_Aminotransferase': aspartate_aminotransferase,
            'Total_Protiens': total_protiens,
            'Albumin': albumin,
            'Albumin_and_Globulin_Ratio': albumin_globulin_ratio
        }])

        X_input = prepare_input(input_df)
        pred = classifier.predict(X_input)
        pred_proba = classifier.predict_proba(X_input)[:, 1] if hasattr(classifier, "predict_proba") else None

        # 🎯 Stylish result display
        st.subheader("🎯 Prediction Result")

        if pred[0] == 1:
            st.markdown(
                f"""
                <div style='background-color:#ffe6e6;padding:20px;border-radius:12px;text-align:center'>
                    <h3 style='color:#cc0000;'>⚠️ Liver Disease Detected</h3>
                    <p style='font-size:18px;'>This patient is <b>likely to have Liver Disease</b>.</p>
                    <p><b>Probability:</b> {pred_proba[0]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style='background-color:#e6ffe6;padding:20px;border-radius:12px;text-align:center'>
                    <h3 style='color:#007700;'>✅ No Liver Disease Detected</h3>
                    <p style='font-size:18px;'>This patient is <b>unlikely to have Liver Disease</b>.</p>
                    <p><b>Probability:</b> {pred_proba[0]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

      except Exception as e:
        st.error(f"Prediction failed: {e}")

            # ----------------------------
               # Option B: Batch Prediction
           # ----------------------------
    st.subheader("📂 Batch Prediction (CSV Upload)")
    batch_file = st.file_uploader("Upload a CSV containing patient data", type=['csv'], key='batch_upload_liver')

    if batch_file is not None:
      try:
        batch_df = pd.read_csv(batch_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        if st.button("Run batch prediction"):
            try:
                X_batch = prepare_input(batch_df)
                preds = classifier.predict(X_batch)
                probs = classifier.predict_proba(X_batch)[:, 1] if hasattr(classifier, "predict_proba") else None

                batch_results = batch_df.copy()
                batch_results['Predicted_Status'] = ['Liver Disease' if p == 1 else 'No Disease' for p in preds]
                if probs is not None:
                    batch_results['Probability'] = probs

                st.success("✅ Batch prediction completed successfully!")
                st.dataframe(batch_results.head())

                csv = batch_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name='liver_predictions.csv',
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
      except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

    st.markdown("---")
    st.caption("🧩 **Note:** Ensure the uploaded CSV has the same feature columns used during training.")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   # page6 -  Developer Info
 
elif page == "Developer Info":
    st.header("Developer Info")
    st.markdown("""
    **Developed by:** T RENUGADEVI 

    **Course:** Data Science                        
    **Skills:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, EDA, Machine Learning Model Development, Streamlit""", True)

    st.snow()