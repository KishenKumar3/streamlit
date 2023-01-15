import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from streamlit_folium import folium_static
import folium
import os
from PIL import Image
import smtplib as s
import pythoncom
from email.message import EmailMessage
import ssl


img = Image.open("EmailPic.png")

main_container = st.container()

main_container.title("Laundry Dataset Modelling")
option = st.sidebar.selectbox("Choose the technique or feature: ", ['Feature Selection','Association Rule Mining','Classification','Regression','Clustering', 'Email'])

if option == "Clustering":
    st.header("K-Means Clustering")
    st.write("This is a scatterplot for Monthly Salary versus Total Sum of Loan")

    st.image('sns plot.png')

    st.write("Elbow method used to determine the approriate number of clusters")

    st.image('elbow.png')

    option = st.selectbox(
        'Select the number of Clusters for K-Means Clustering',
        ('1', '2', '3'))

    st.write('You selected:', option)

    if option == '1':
    
        st.write("This is a scatterplot for Monthly Salary versus Total Sum of Loan with k = 1")
        st.image('k1.png')

    elif option == '2':
    
        st.write("This is a scatterplot for Monthly Salary versus Total Sum of Loan with k = 2")
        st.image('k2.png')

    elif option == '3':
    
        st.write("This is a scatterplot for Monthly Salary versus Total Sum of Loan with k = 3")
        st.image('k3.png')

elif option == "Classification":
    st.header("Naive Bayes Classifier")                                           

    st.subheader("Model Accuracy Before Hyper-Parameter Tuning")                                           
    st.write("Model Accuracy score = 0.72340")
    st.write("Training-set accuracy score = 0.7176")
    
    st.subheader("Confusion Matrix Before Hyper-Parameter Tuning")                                           

    st.write("""Confusion Matrix

    [[337  30]
    [100   3]]

    True Positives(TP) =  337

    True Negatives(TN) =  3

    False Positives(FP) =  30

    False Negatives(FN) =  100""")


    st.text("")
    st.text("""The confusion matrix shows 337 + 3 = 340 correct predictions and 30 + 100 = 130 incorrect predictions.

    In this case, we have

    True Positives (Actual Positive:1 and Predict Positive:1) - 337
    True Negatives (Actual Negative:0 and Predict Negative:0) - 3
    False Positives (Actual Negative:0 but Predict Positive:1) - 30 (Type I error)
    False Negatives (Actual Positive:1 but Predict Negative:0) - 100 (Type II error)""")

    st.write("Visualize the confusion Matrix through heatmap")
    st.image('errHeat.png')

    st.subheader("Best Parameter Changes")

    # Write your parameter changes here


    st.subheader("Model Accuracy After Hyper-Parameter Tuning")                                           
    st.write("Model Accuracy score = 0.72340")
    st.write("Training-set accuracy score = 0.7176")
    
    st.subheader("Confusion Matrix After Hyper-Parameter Tuning")                                           

    st.write("""Confusion Matrix

    [[337  30]
    [100   3]]

    True Positives(TP) =  337

    True Negatives(TN) =  3

    False Positives(FP) =  30

    False Negatives(FN) =  100""")


    st.text("")
    st.text("""The confusion matrix shows 337 + 3 = 340 correct predictions and 30 + 100 = 130 incorrect predictions.

    In this case, we have

    True Positives (Actual Positive:1 and Predict Positive:1) - 337
    True Negatives (Actual Negative:0 and Predict Negative:0) - 3
    False Positives (Actual Negative:0 but Predict Positive:1) - 30 (Type I error)
    False Negatives (Actual Positive:1 but Predict Negative:0) - 100 (Type II error)""")

    st.write("Visualize the confusion Matrix through heatmap")
    st.image('errHeat.png')




    st.header("Decision Tree Classifier")                                           

    st.subheader("Model Accuracy Before Hyper-Parameter Tuning")                                           
    st.write("Model Accuracy score = 0.7809")
    st.write("Training-set accuracy score = 0.7457")

    st.subheader("Confusion matrix Before Hyper-Parameter Tuning")                                           

    st.write("""Confusion Matrix

    [[367   0]
    [103   0]]

    True Positives(TP) =  367

    True Negatives(TN) =  0

    False Positives(FP) =  0

    False Negatives(FN) =  103""")

    st.text("")
    st.text("""The confusion matrix shows 367 + 0 = 367 correct predictions and 103 + 0 = 103 incorrect predictions.

    In this case, we have

    True Positives (Actual Positive:1 and Predict Positive:1) - 367
    True Negatives (Actual Negative:0 and Predict Negative:0) - 0
    False Positives (Actual Negative:0 but Predict Positive:1) - 0 (Type I error)
    False Negatives (Actual Positive:1 but Predict Negative:0) - 103 (Type II error)""")


    st.write("Visualize the confusion Matrix through heatmap")
    st.image('cm2.png')

    st.subheader("Visualize decision-tree Before Hyper-Parameter Tuning")                                           
    st.image('dt.png')

    st.subheader("Best Parameter Changes")

    # Write your paramter changes here

    st.subheader("Model Accuracy After Hyper-Parameter Tuning")                                           
    st.write("Model Accuracy score = 0.7809")
    st.write("Training-set accuracy score = 0.7457")

    st.subheader("Confusion matrix After Hyper-Parameter Tuning")                                           

    st.write("""Confusion Matrix

    [[367   0]
    [103   0]]

    True Positives(TP) =  367

    True Negatives(TN) =  0

    False Positives(FP) =  0

    False Negatives(FN) =  103""")

    st.text("")
    st.text("""The confusion matrix shows 367 + 0 = 367 correct predictions and 103 + 0 = 103 incorrect predictions.

    In this case, we have

    True Positives (Actual Positive:1 and Predict Positive:1) - 367
    True Negatives (Actual Negative:0 and Predict Negative:0) - 0
    False Positives (Actual Negative:0 but Predict Positive:1) - 0 (Type I error)
    False Negatives (Actual Positive:1 but Predict Negative:0) - 103 (Type II error)""")


    st.write("Visualize the confusion Matrix through heatmap")
    st.image('cm2.png')

    st.subheader("Visualize decision-tree After Hyper-Parameter Tuning")                                           
    st.image('dt.png')

    st.subheader("Comparison between Classifying Models")                                           



    data = {'Models':  ['Naive Bayes', 'Decision tree'],
            'Accuracy on Train': ['0.7176', '0.7457'],
            'Accuracy on Test': ['0.72340', '0.7809']
        }


    st.write(pd.DataFrame(data))

    st.subheader("Conclusion")                                           
    st.write("""The accuracy on the Decision Tree is much better than compared to the Naive Bayes Classifier.
            The impact on the parameter change was quite big when I changed the random state and the test data size for the Naive Bayes Classifier. As for the decision tree, I made a comparison between choosing the critirion Entropy and Gini and i concluded that entropy had a better accuracy on test.
            The Decision Tree classifies the class better than Naive Bayes as it has better accuracy and lacks errors as compared to Naive Bayes.""")




elif option == "Feature Selection":
    st.header("Chi-Squared Test")

    st.subheader("Top 10 Features")
    st.write("""
    1) With_Kids_no
    2) With_Kids_yes
    3) Kids_Category_no_kids
    4) Kids_Category_toddler 
    5) Kids_Category_young 
    6) Attire_casual
    7) Attire_traditional 
    8) shirt_type_long sleeve 
    9) Spectacles_no
    10) Spectacles_yes
    """)

    st.subheader("Bottom 10 Features")
    st.write("""
    1) With_Kids_no
    2) With_Kids_yes
    3) Kids_Category_no_kids
    4) Kids_Category_toddler 
    5) Kids_Category_young 
    6) Attire_casual
    7) Attire_traditional 
    8) shirt_type_long sleeve 
    9) Spectacles_no
    10) Spectacles_yes
    """)

    st.subheader("Visualization of feature scores")
    st.image("feature scores_chi.png")

    st.text("")

    st.header("Boruta")

    st.subheader("Top 10 Features")
    st.image("top10features_Boruta.png")

    st.subheader("Bottom 10 Features")
    st.image("bottom10features_Boruta.PNG")

    st.subheader("Visualization of feature scores")
    st.image("feature scores_boruta.png")

    st.text("")

    st.header("Combining Features using Intersection Method")

    st.subheader("Top 10 optimal feature set from the combination of Chi-Squared Test and Boruta Method")
    st.write("""
    1) With_Kids_no
    2) With_Kids_yes
    3) Kids_Category_no_kids
    4) Kids_Category_toddler 
    5) Kids_Category_young 
    6) Attire_casual
    7) Attire_traditional 
    8) shirt_type_long sleeve 
    9) Spectacles_no
    10) Spectacles_yes
    """)

# Email Sender Address: mackishenkumar3@gmail.com
# Email Sender Password: fzajkcdrrsdsgwup

elif option == "Email":
    st.title("Email Sender Feature")
    st.write("Build with Streamlit and Python")
    st.image(img, width=200)
    email_sender = st.text_input("Enter User Email")
    password = st.text_input("Enter User Password", type='password')
    email_receiver = st.text_input("Enter Receiver Email")
    subject = st.text_input("Your email subject")
    body = st.text_area("Your email")
    em = EmailMessage()
    if st.button("Send email"):
        try:
            em['From'] = email_sender
            em['To'] = email_receiver
            em['Subject'] = subject
            em.set_content(body)
            context = ssl.create_default_context()
            with s.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                smtp.login(email_sender, password)
                smtp.sendmail(email_sender, email_receiver, em.as_string())
                st.success("Email Send Succesfully")

        except Exception as e:
            if email_sender=="":
                st.error("Please fill User Email Field")
            elif password == "":
                st.error("Please fill Password Field!")
            elif email_receiver == "":
                st.error("Please fill Receiver Email Field")
            else:
                a=os.system("ping www.google.com")
                if a==1:
                    st.error("Please check your internet connection")
                else:
                    st.error("Wrong email or password!")

    else:
        pass



            



            











