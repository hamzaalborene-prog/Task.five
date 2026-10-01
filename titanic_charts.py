# استيراد المكتبات اللازمة
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# تحميل بيانات Titanic
تيتانيك = sns.load_dataset("titanic")

# تجهيز البيانات
بيانات_العمر = تيتانيك.dropna(subset=["age"])
بيانات_الأجرة = تيتانيك.dropna(subset=["age", "fare"])
بيانات_الارتباط = تيتانيك.select_dtypes(include="number")

# الجزء الأول: Matplotlib

# 1. توزيع أعمار الركاب
plt.figure(figsize=(8, 5))
plt.hist(بيانات_العمر["age"], bins=20, edgecolor="black")
plt.title("توزيع أعمار الركاب")
plt.xlabel("العمر")
plt.ylabel("عدد الركاب")
plt.tight_layout()
plt.show()

# 2. عدد الركاب حسب درجة التذكرة
عدد_الركاب = تيتانيك["pclass"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(عدد_الركاب.index.astype(str), عدد_الركاب.values)
plt.title("عدد الركاب حسب درجة التذكرة")
plt.xlabel("درجة التذكرة")
plt.ylabel("عدد الركاب")
plt.tight_layout()
plt.show()

# 3. العلاقة بين العمر وسعر التذكرة
plt.figure(figsize=(8, 5))
plt.scatter(بيانات_الأجرة["age"], بيانات_الأجرة["fare"], alpha=0.5)
plt.title("العلاقة بين العمر وسعر التذكرة")
plt.xlabel("العمر")
plt.ylabel("سعر التذكرة")
plt.tight_layout()
plt.show()

# الجزء الثاني: Seaborn

# 4. خريطة حرارية للارتباط
plt.figure(figsize=(10, 7))
sns.heatmap(بيانات_الارتباط.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("خريطة الارتباط بين الأعمدة الرقمية")
plt.tight_layout()
plt.show()

# 5. توزيع الأعمار حسب درجة التذكرة
plt.figure(figsize=(8, 5))
sns.boxplot(data=بيانات_العمر, x="pclass", y="age")
plt.title("توزيع الأعمار حسب درجة التذكرة")
plt.xlabel("درجة التذكرة")
plt.ylabel("العمر")
plt.tight_layout()
plt.show()

# 6. عدد الركاب حسب الجنس وحالة النجاة
plt.figure(figsize=(8, 5))
sns.countplot(data=تيتانيك, x="sex", hue="survived")
plt.title("عدد الركاب حسب الجنس وحالة النجاة")
plt.xlabel("الجنس")
plt.ylabel("عدد الركاب")
plt.legend(title="النجاة", labels=["لم ينجُ", "نجا"])
plt.tight_layout()
plt.show()

# 7. العلاقات بين ثلاثة أعمدة رقمية
sns.pairplot(تيتانيك[["age", "fare", "sibsp"]].dropna())
plt.suptitle("العلاقات بين العمر والأجرة وعدد الإخوة أو الأزواج", y=1.02)
plt.show()

# الجزء الثالث: Plotly

# 8. عدد الركاب حسب درجة التذكرة
رسم_الدرجة = px.bar(
    تيتانيك.groupby("pclass", as_index=False).size(),
    x="pclass", y="size",
    title="عدد الركاب حسب درجة التذكرة",
    labels={"pclass": "درجة التذكرة", "size": "عدد الركاب"}
)
رسم_الدرجة.show()

# 9. العمر وسعر التذكرة مع التلوين حسب النجاة
رسم_العمر_والأجرة = px.scatter(
    بيانات_الأجرة, x="age", y="fare", color="survived",
    title="العمر وسعر التذكرة حسب حالة النجاة",
    labels={"age": "العمر", "fare": "سعر التذكرة", "survived": "النجاة"},
    hover_data=["sex", "pclass"]
)
رسم_العمر_والأجرة.show()

# 10. توزيع الأعمار حسب الجنس
رسم_توزيع_العمر = px.histogram(
    بيانات_العمر, x="age", color="sex",
    title="توزيع الأعمار حسب الجنس",
    labels={"age": "العمر", "sex": "الجنس"},
    barmode="overlay", opacity=0.65
)
رسم_توزيع_العمر.show()

# حفظ رسم تفاعلي بصيغة HTML
رسم_العمر_والأجرة.write_html("titanic_interactive_chart.html")

# الجزء الرابع: أسئلة تحليل الأعمال

# 11. مقارنة أسعار التذاكر حسب الدرجة
plt.figure(figsize=(8, 5))
sns.boxplot(data=تيتانيك, x="pclass", y="fare")
plt.title("مقارنة أسعار التذاكر حسب درجة التذكرة")
plt.xlabel("درجة التذكرة")
plt.ylabel("سعر التذكرة")
plt.tight_layout()
plt.show()
# اخترنا المخطط الصندوقي لأنه يوضح الوسيط وانتشار الأسعار والقيم المتطرفة.
# التفسير: تختلف أسعار التذاكر بين درجات السفر، وتظهر الأسعار الأعلى غالبًا في الدرجة الأولى.

# 12. حساب النسبة العامة للنجاة
نسبة_النجاة = تيتانيك["survived"].mean() * 100
print(f"نسبة النجاة العامة: {نسبة_النجاة:.2f}%")
# النسبة المئوية تلخص حصة الناجين من جميع الركاب.
# التفسير: تمثل النسبة الناتجة الجزء الذي نجا من الركاب في هذه البيانات.

# 13. حجم العائلة وعلاقته بالنجاة
تيتانيك["حجم_العائلة"] = تيتانيك["sibsp"] + تيتانيك["parch"] + 1
معدل_النجاة_حسب_العائلة = تيتانيك.groupby("حجم_العائلة", as_index=False)["survived"].mean()
معدل_النجاة_حسب_العائلة["survived"] *= 100

plt.figure(figsize=(9, 5))
sns.barplot(data=معدل_النجاة_حسب_العائلة, x="حجم_العائلة", y="survived", color="steelblue")
plt.title("نسبة النجاة حسب حجم العائلة")
plt.xlabel("حجم العائلة")
plt.ylabel("نسبة النجاة (%)")
plt.tight_layout()
plt.show()
# اخترنا الأعمدة لتسهيل مقارنة نسب النجاة بين أحجام العائلات.
# التفسير: يختلف معدل النجاة باختلاف حجم العائلة، ولا يعني ذلك أن حجم العائلة وحده سبب النجاة.

# 14. متوسط سعر التذكرة حسب ميناء الصعود
متوسط_الأجرة = تيتانيك.groupby("embarked", as_index=False)["fare"].mean().dropna()
plt.figure(figsize=(8, 5))
sns.barplot(data=متوسط_الأجرة, x="embarked", y="fare", color="cornflowerblue")
plt.title("متوسط سعر التذكرة حسب ميناء الصعود")
plt.xlabel("ميناء الصعود")
plt.ylabel("متوسط سعر التذكرة")
plt.tight_layout()
plt.show()
# اخترنا الأعمدة لمقارنة متوسطات الأسعار بين الموانئ.
# التفسير: يوضح الرسم اختلاف متوسط سعر التذكرة بين الموانئ، ولا يثبت أن الميناء سبب الاختلاف.
