import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from streamlit_folium import folium_static
import folium
import os
from PIL import Image
import smtplib as s
from email.message import EmailMessage
import ssl
import pickle

img = Image.open("EmailPic.png")

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def basket_size_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'small'
    else:
        return 'big'


main_container = st.container()

main_container.title("Laundry Dataset Modelling")
option = st.sidebar.selectbox("Choose the technique or feature: ", ['Exploratory Data Analysis', 'Feature Selection','Basket Size Predictive Model','Clustering', 'Email'])

if option == 'Exploratory Data Analysis':
    st.header('Exploratory Data Analysis')
    st.subheader('Provided Dataset')
    dataset = pd.read_csv('dataset.csv')
    df = pd.DataFrame(dataset)
    df
    st.subheader("Checking duplicated values")
    st.image("DuplicatedRows.png")
    st.subheader("Cleaning Noisy Data")
    st.image('NoisyData.png')
    st.subheader('Cleaning Missing Values')
    missingOption = st.selectbox("Missing Values for column: ", ['Categorical Column', 'Numerical Column'])
    if missingOption == 'Categorical Column':
        st.write('Missing Values for categorical columns')
        st.image('categoricalColumnsMissing.png')
    elif missingOption == 'Numerical Column':
        st.write('Missing Values for Numerical columns')
        st.image('numericalColumnsMissing.png')
    st.write("Used 'KNN Imputer' to treat missing values in numerical columns and 'Mode' value for each column in categorical columns")
    st.subheader('Checking skewness')
    st.image('skewness.png')
    st.write('A skewness value greater than 1 or less than -1 indicates a highly skewed distribution. Since the values for all those numerical columns are between those values, this indicates that it is fairly symmetrical.')
    st.subheader("Checking and correcting data types of needed columns")
    st.image('DataTypes.png')
    st.subheader('Checking Outliers')
    st.image('outliers.png')
    st.write("As we can see, the columns don't have outliers. This is because, the max row for all columns is not much different in value from their mean. The mean is sensitive to outliers, but the fact the mean is not so far apart compared to the max value indicates there aren't outliers. As we explore the data using additional methods, we can further proof this.")
    st.subheader('Cleaned Dataset')
    dataset2 = pd.read_csv('datasetFixed.csv')
    df2 = pd.DataFrame(dataset2)
    df2
    st.subheader('Distribution of attires in a pie chart')
    st.image('piechart.png')
    st.write('Customers wore casual attire more than other types of attires.')
    st.subheader('Relationship between features')
    st.image('relationship.png')
    st.subheader('Distribution of age among people differed by their race and gender')
    st.image('ageDistribution.png')
    st.write('We can tell most of the races are above the ages of 30.')
    st.subheader('Total expenditure of each race of customers in laundry shop')
    st.image('expenditure.png')
    st.write('From the Boxplot above, we can see chinese and foreigners happen to have the biggest IQR compared to other races, which means its more spread out.')
    st.subheader('Correlation Between Features')
    st.image('correlation.png')
    st.write('None of the values from the dataset seem to correlate with each other.')
    st.subheader('Data Imbalance Treatment')
    balancedOption = st.selectbox("Data Imbalance Treatment: ", ['Before balancing the target class', 'Oversampling Minority Class (SMOTE)'])
    if balancedOption == "Before balancing the target class":
        st.image('beforebalanced.png')
        st.write("From this figure, we can say that the data is imbalanced since 'small' values is quite lower compared to the 'big' values' in the target class.")
    elif balancedOption == "Oversampling Minority Class (SMOTE)":
        st.image('afterbalanced.png')
        st.write('The target class values count is almost the same.')

    st.subheader('External Dataset')
    dataset3 = pd.read_csv('weather_dataset.csv')
    df3 = pd.DataFrame(dataset3)
    df3
    st.write("Sucessfully webscraped 'weather' and 'town' attibutes from a weather data API.")

    st.subheader("Relationship between 'basket size' and 'race'")
    st.image('raceBasket.png')
    st.write('From the barplot we can see all races would go for big basket sizes than compared to smaller ones.')

    st.subheader("Impact of weather on sales")
    st.image('weather.png')
    st.write('During almost to rain weathers, we can see sales increase and decrease slightly. This may be due to people not being able to dry their clothes at home due to the lack of sunny time. So they opt for laundry shops dryers.')

    st.subheader("Types of customers that will likely to choose Washer No. 4 and Dryer No. 10")
    st.image('washer.png')
    st.write('From the bar plot we can tell females tend to use washer no.4 and Dryer no.10 more. This could be because they have more clothes as those machines are much bigger.')



if option == "Clustering":
    st.header("K-Means Clustering")

    st.write("Elbow method used to determine the approriate number of clusters")

    st.image('elbowplot.png')

    option = st.selectbox(
        'Select the number of Clusters for K-Means Clustering',
        ('1', '2', '3', '4', '5'))

    st.write('You selected:', option)

    if option == '1':
        st.write("This is a scatterplot for Age Range versus Total Money Spent(RM) with k = 1")
        st.image('k1.png')

    elif option == '2':
        st.write("This is a scatterplot for Age Range versus Total Money Spent(RM) with k = 2")
        st.image('k2.png')

    elif option == '3':
        st.write("This is a scatterplot for Age Range versus Total Money Spent(RM) with k = 3")
        st.image('k3.png')

    elif option == "4":
        st.write("This is a scatterplot for Age Range versus Total Money Spent(RM) with k = 4")
        st.image('k4.png')

    elif option == "5":
        st.write("This is a scatterplot for Age Range versus Total Money Spent(RM) with k = 5")
        st.image('k5.png')

elif option == "Basket Size Predictive Model":
    st.title("Basket Size Predictive Model")

    Spectacles_No = st.text_input("Did the customer wear spectacles? (Enter '0' for Yes and '1' for No)")
    if Spectacles_No == "":
        st.error("This field is required.")
    else:
        Spectacles_No = int(Spectacles_No)
        if not (Spectacles_No == 0 or Spectacles_No == 1):
            st.error("Invalid input, please enter either '0' or '1'")

    ShirtType_LongSleeve = st.text_input("Did the customer wear long sleeve shirt? (Enter '0' for No and '1' for Yes)")
    if ShirtType_LongSleeve == "":
        st.error("This field is required.")
    else:
        ShirtType_LongSleeve = int(ShirtType_LongSleeve)
        if not (ShirtType_LongSleeve == 0 or ShirtType_LongSleeve == 1):
            st.error("Invalid input, please enter either '0' or '1'")

    Attire_Traditional = st.text_input("Did the customer wear traditional attire? (Enter '0' for No and '1' for Yes)")
    if Attire_Traditional == "":
        st.error("This field is required.")
    else:
        Attire_Traditional = int(Attire_Traditional)
        if not (Attire_Traditional == 0 or Attire_Traditional == 1):
            st.error("Invalid input, please enter either '0' or '1'")

    Kids_Category_No_Kids = st.text_input("Did the customer have kids? (Enter '0' for Others and '1' for No Kids)")
    if Kids_Category_No_Kids == "":
        st.error("This field is required.")
    else:
        Kids_Category_No_Kids = int(Kids_Category_No_Kids)
        if not (Kids_Category_No_Kids == 0 or Kids_Category_No_Kids == 1):
            st.error("Invalid input, please enter either '0' or '1'")

    With_Kids_Yes = st.text_input("Did the customer come with kids? (Enter '1' for Yes and '0' for No)")
    if With_Kids_Yes == "":
        st.error("This field is required.")
    else:
        With_Kids_Yes = int(With_Kids_Yes)
        if not (With_Kids_Yes == 0 or With_Kids_Yes == 1):
            st.error("Invalid input, please enter either '0' or '1'")

    Attire_Casual = 0
    With_Kids_No = 0

    if Attire_Traditional == 0:
        Attire_Casual == 1
    elif Attire_Traditional == 1:
        Attire_Casual == 0

    if With_Kids_Yes == 0:
        With_Kids_No == 1
    elif With_Kids_Yes == 1:
        With_Kids_No == 0

    target = ''

    if st.button('Basket Size Prediction Result'):
        target = basket_size_prediction([Spectacles_No, ShirtType_LongSleeve, Attire_Traditional, Attire_Casual, Kids_Category_No_Kids, With_Kids_Yes, With_Kids_No])

    st.success(target)




elif option == "Feature Selection":
    st.header("Chi-Squared Test")

    foption = st.selectbox(
        'Select top 10 features or bottom 10 features from chi-squared test',
        ('Top 10 Features', 'Bottom 10 Features'))

    if foption == 'Top 10 Features':
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

    elif foption == 'Bottom 10 Features':
        st.subheader("Bottom 10 Features")
        st.write("""
    1) buyDrinks
    2) Washer_No
    3) Dryer_No
    4) TotalSpent_RM 
    5) latitude 
    6) Race_chinese
    7) Gender_female 
    8) Race_indian 
    9) Body_Size_moderate
    10) Gender_male
    """)

    st.subheader("Visualization of feature scores")
    st.image("feature scores_chi.png")

    st.text("")

    st.header("Boruta")
    boption = st.selectbox(
        'Select top 10 features or bottom 10 features from Boruta method',
        ('Top 10 Features', 'Bottom 10 Features'))

    if boption == 'Top 10 Features':
        st.subheader("Top 10 Features")
        st.image("top10features_Boruta.png")

    elif boption == 'Bottom 10 Features':
        st.subheader("Bottom 10 Features")
        st.image("bottom10features_Boruta.png")

    st.subheader("Visualization of feature scores")
    st.image("feature scores_boruta.png")

    st.text("")

    st.header("Combining Features using Intersection Method")

    st.subheader("Optimal feature set from the combination of Chi-Squared Test and Boruta Method")
    st.write("""
            1) Spectacles_no

            2) shirt_type_long sleeve

            3) Attire_traditional

            4) Attire_casual

            5) Kids_Category_no_kids

            6) With_Kids_yes

            7) With_Kids_no
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



            



            











