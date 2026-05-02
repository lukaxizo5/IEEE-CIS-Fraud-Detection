# IEEE-CIS-Fraud-Detection

## პროექტის მოკლე აღწერა
ეს პროექტი არის supervised learning-ის ერთ-ერთი მაგალითი, რომელშიც ჩვენი მიზანია დავადგინოთ ტრანზაქცია ყალბია თუ არა.
პრობლემა წარმოადგენს კლასიფიკაციის ამოცანას.

### შეფასების მეტრიკა
მოდელი ფასდება AUROC (Area Under ROC Curve) მეტრიკით.

## ჩემი მიდგომა
პრობლემის გადასაჭრელად გამოვიყენე შემდეგი workflow:
1. **EDA** - მონაცემების შესწავლა და პრობლემების იდენტიფიცირება
2. **Data Cleaning** - missing values დამუშავება და ცვლადების encoding
3. **Feature Engineering** - ახალი ინფორმაციული ცვლადების შექმნა
4. **Feature Selection** - უსარგებლო ცვლადების ამოღება და სწორი ცვლადების შერჩევა მოდელის დასატრენინგებლად
5. **Model Training** - baseline მოდელებიდან advanced მოდელებამდე + HPO (Hyperparameter Optimization)

## რეპოზიტორიის სტრუქტურა
```
IEEE-CIS-Fraud-Detection/
├── images/                                      # გრაფიკები README-თვის
├── model_experiment_{model_architecture}.ipynb  # თითოეული მოდელის არქიტექტურისთვის მთავარი სამუშაო ფაილი მოდელის დასატრენინგებლად
├── model_inference.ipynb                        # საბოლოო პროგნოზი და submission
└── README.md                                    # პროექტის დოკუმენტაცია
```

## Exploratory Data Analysis (EDA)

### მონაცემების სტრუქტურა
- train_df-ს აქვს 590540 სტრიქონი და 434 სვეტი.
- 403 numerical სვეტი, 31 categorical სვეტი.

### Target (isFraud) განაწილება
![Target Distribution](images/eda/target_distribution.png)
მონაცემები არაბალანსირებულია - მხოლოდ **3.5%** ტრანზაქციაა fraudulent.
ეს ნიშნავს, რომ Accuracy უსარგებლო მეტრიკაა, რადგან მოდელი რომელიც ყველაფერზე იტყვის რომ not fraud არის, 96.5% შემთხვევებში სწორი იქნება.
სწორედ ამიტომაა მეტრიკად არჩეული AUROC, რომელიც აფასებს მოდელის უნარს გაარჩიოს ორი კლასი ერთმანეთისგან პროპორციის მიუხედავად.

### Missing Values
![Missing Values](images/eda/missing_values.png)
Missing values-ების ანალიზმა აჩვენა, რომ სვეტების დიდი ნაწილი შეიცავს 
გამოტოვებულ მნიშვნელობებს. გრაფიკი გვიჩვენებს რამდენ ცვლადს აქვს კონკრეტულ 
ზღვარზე მეტი missing values:

- 70%-ზე მეტი missing: **208 ცვლადი**
- 80%-ზე მეტი missing: **74 ცვლადი**
- 90%-ზე მეტი missing: **12 ცვლადი**

70%-დან 80%-მდე გადასვლისას ყველაზე მკვეთრი კლებაა — 208-დან 74-მდე. 
სწორედ ეს არის ბუნებრივი ზღვარი, ამიტომ Cleaning ეტაპზე ამოვიღებ ყველა 
ცვლადს, რომელსაც **80%-ზე მეტი** missing values აქვს.

### Transaction Amount Analysis
![Transaction Amount](images/eda/transaction_amount.png)
ტრანზაქციის თანხის განაწილება 2 კლასს შორის მსგავსია - მედიანებსა და საშუალოებს შორის არაა დიდი განსხვავება.
ეს ნიშნავს, რომ `TransactionAmt` მარტო სუსტი სიგნალია, თუმცა კომბინაციაში შეიძლება გამოგვადგეს.

### Categorical Feature Fraud Rates
![Categorical Fraud Rates](images/eda/categorical_fraud_rates.png)
დავაკვირდი რამდენიმე კატეგორიულ ცვლადს და მათ შორის target-ის განაწილებებს.

| ცვლადი | დაკვირვება                                                                  |
|---|-----------------------------------------------------------------------------|
| `ProductCD` | C კატეგორიას აქვს ~12% fraud rate — ყველაზე მაღალი                          |
| `card4` | Discover ბარათს აქვს ~8% fraud rate — Visa/Mastercard-ზე გაცილებით მაღალი   |
| `card6` | Credit ბარათები (~6.5%) დაახლოებით 2.5-ჯერ უფრო საეჭვოა ვიდრე Debit (~2.5%) |
| `P_emaildomain` | გარკვეული დომენები აღწევს ~40% fraud rate-ს                                 |
| `R_emaildomain` | გარკვეული დომენები აღწევს ~80-90% fraud rate-ს — ძალიან ძლიერი სიგნალი      |


### Transaction Time Analysis
![Transaction Time](images/eda/transaction_time.png)
დღის საათი მნიშვნელოვანი სიგნალია:
- **4-10** საათებში ფიქსირდება ყველაზე მაღალი fraud rate.

სავარაუდოდ, ეს იმითაა გამოწვეული, რომ 4-10 საათებში ხალხს უმეტესად სძინავს და საეჭვო ტრანზაქციებს ვერ ამჩნევენ.

### Correlations
![Correlations](images/eda/correlations.png)
კორელაციების ანალიზი ჩავატარე, თუმცა შენიღბულია მონაცემები, ამიტომ ეს ამ ეტაპზე დიდად ინფორმაციული არაა.
