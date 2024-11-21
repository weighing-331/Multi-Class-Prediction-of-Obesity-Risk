import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib


# Load the trained model
#model_path = "lgbm_model.pkl"

#lgbm_model = joblib.load(model_path)
lgbm_model = joblib.load("C:/Users/88693/Downloads/lgbm_model.pkl")

# Initialize the main window
root = tk.Tk()
root.title("肥胖風險預測")
root.geometry("600x800")

# Create a scrollable frame
canvas = tk.Canvas(root)
scrollable_frame = ttk.Frame(canvas)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Dictionary to hold the controls
controls = {}

# Helper function to create a scale widget
def create_scale(frame, label_text, from_, to, resolution):
    label_var = tk.StringVar()
    label_var.set(f"{label_text}: {from_}")
    ttk.Label(frame, textvariable=label_var).pack(side=tk.LEFT)
    scale = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, length=200, command=lambda val: label_var.set(f"{label_text}: {float(val):.1f}"))
    scale.set(from_)
    scale.pack(side=tk.LEFT, padx=5)
    return scale

# Helper function to create entry widget
def create_entry(frame, label_text):
    ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
    entry = ttk.Entry(frame)
    entry.pack(side=tk.LEFT, padx=5, fill='x', expand=True)
    return entry

# Adding the features with their respective input methods

# 年齡
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
age_entry = create_entry(frame, "年齡")
controls["Age"] = age_entry

# 身高
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
height_entry = create_entry(frame, "身高")
controls["Height"] = height_entry

# 體重
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
weight_entry = create_entry(frame, "體重")
controls["Weight"] = weight_entry

# 蔬菜攝取頻率 (FCVC)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
fcvc_scale = create_scale(frame, "蔬菜攝取頻率 (FCVC)", 0, 3, 0.1)
controls["FCVC"] = fcvc_scale

# 主餐數量 (NCP)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ncp_scale = create_scale(frame, "主餐數量 (NCP)", 1, 3, 1)
controls["NCP"] = ncp_scale

# 每日飲水量 (CH2O)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ch2o_scale = create_scale(frame, "每日飲水量 (CH2O)", 0, 3, 0.1)
controls["CH2O"] = ch2o_scale

# 運動頻率 (FAF)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
faf_scale = create_scale(frame, "運動頻率 (FAF)", 0, 5, 0.1)
controls["FAF"] = faf_scale

# 使用技術設備時間 (TUE)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
tue_scale = create_scale(frame, "使用技術設備時間 (TUE)", 0, 24, 0.1)
controls["TUE"] = tue_scale

# 性別
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ttk.Label(frame, text="性別").pack(side=tk.LEFT)
gender_cb = ttk.Combobox(frame, values=["Male", "Female"], state="readonly")
gender_cb.pack(side=tk.LEFT, expand=True, fill='x')
controls["Gender"] = gender_cb

# 家族肥胖史
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ttk.Label(frame, text="家族肥胖史").pack(side=tk.LEFT)
fh_cb = ttk.Combobox(frame, values=["Yes", "No"], state="readonly")
fh_cb.pack(side=tk.LEFT, expand=True, fill='x')
controls["family_history_with_overweight"] = fh_cb

# 高熱量食物攝取頻率 (FAVC)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ttk.Label(frame, text="高熱量食物攝取頻率 (FAVC)").pack(side=tk.LEFT)
favc_cb = ttk.Combobox(frame, values=["Yes", "No"], state="readonly")
favc_cb.pack(side=tk.LEFT, expand=True, fill='x')
controls["FAVC"] = favc_cb

# 餐間食物攝取 (CAEC)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ttk.Label(frame, text="餐間食物攝取 (CAEC)").pack(side=tk.LEFT)
caec_cb = ttk.Combobox(frame, values=["No", "Sometimes", "Frequently", "Always"], state="readonly")
caec_cb.pack(side=tk.LEFT, expand=True, fill='x')
controls["CAEC"] = caec_cb

# 吸煙 (SMOKE)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ttk.Label(frame, text="吸煙 (SMOKE)").pack(side=tk.LEFT)
smoke_cb = ttk.Combobox(frame, values=["Yes", "No"], state="readonly")
smoke_cb.pack(side=tk.LEFT, expand=True, fill='x')
controls["SMOKE"] = smoke_cb

# 卡路里攝入監控 (SCC)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ttk.Label(frame, text="卡路里攝入監控 (SCC)").pack(side=tk.LEFT)
scc_cb = ttk.Combobox(frame, values=["Yes", "No"], state="readonly")
scc_cb.pack(side=tk.LEFT, expand=True, fill='x')
controls["SCC"] = scc_cb

# 飲酒頻率 (CALC)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ttk.Label(frame, text="飲酒頻率 (CALC)").pack(side=tk.LEFT)
calc_cb = ttk.Combobox(frame, values=["No", "Sometimes", "Frequently", "Always"], state="readonly")
calc_cb.pack(side=tk.LEFT, expand=True, fill='x')
controls["CALC"] = calc_cb

# 交通工具 (MTRANS)
frame = ttk.Frame(scrollable_frame)
frame.pack(padx=10, pady=5, fill='x')
ttk.Label(frame, text="交通工具 (MTRANS)").pack(side=tk.LEFT)
mtrans_cb = ttk.Combobox(frame, values=["Walking", "Bike", "Motorbike", "Public Transportation", "Automobile"], state="readonly")
mtrans_cb.pack(side=tk.LEFT, expand=True, fill='x')
controls["MTRANS"] = mtrans_cb

# Function to gather inputs, convert to correct format, and predict
def predict():
    try:
        input_data = []

        # Gender (One-hot encoding)
        gender = controls["Gender"].get()
        input_data.append(1 if gender == "Male" else 0)  # Gender_Male

        # Family History with Overweight (One-hot encoding)
        fh = controls["family_history_with_overweight"].get()
        input_data.append(1 if fh == "Yes" else 0)  # family_history_with_overweight_yes

        # High Caloric Food Consumption (One-hot encoding)
        favc = controls["FAVC"].get()
        input_data.append(1 if favc == "Yes" else 0)  # FAVC_yes

        # Smoking (One-hot encoding)
        smoke = controls["SMOKE"].get()
        input_data.append(1 if smoke == "Yes" else 0)  # SMOKE_yes

        # Calorie Consumption Monitoring (One-hot encoding)
        scc = controls["SCC"].get()
        input_data.append(1 if scc == "Yes" else 0)  # SCC_yes

        # Eating Between Meals (One-hot encoding)
        caec = controls["CAEC"].get()
        caec_no = 1 if caec == "No" else 0
        caec_sometimes = 1 if caec == "Sometimes" else 0
        caec_frequently = 1 if caec == "Frequently" else 0
        input_data.extend([caec_frequently, caec_sometimes, caec_no])

        # Alcohol Consumption (One-hot encoding)
        calc = controls["CALC"].get()
        calc_no = 1 if calc == "No" else 0
        calc_sometimes = 1 if calc == "Sometimes" else 0
        calc_frequently = 1 if calc == "Frequently" else 0
        input_data.extend([calc_frequently, calc_sometimes, calc_no])

        # Transportation (One-hot encoding)
        mtrans = controls["MTRANS"].get()
        mtrans_bike = 1 if mtrans == "Bike" else 0
        mtrans_motorbike = 1 if mtrans == "Motorbike" else 0
        mtrans_public_transportation = 1 if mtrans == "Public Transportation" else 0
        mtrans_walking = 1 if mtrans == "Walking" else 0
        input_data.extend([mtrans_bike, mtrans_motorbike, mtrans_public_transportation, mtrans_walking])

        # Other features (Continuous or ordinal)
        input_data.append(float(controls["Age"].get()))
        input_data.append(float(controls["Height"].get()))
        input_data.append(float(controls["Weight"].get()))
        input_data.append(float(controls["FCVC"].get()))
        input_data.append(float(controls["NCP"].get()))
        input_data.append(float(controls["CH2O"].get()))
        input_data.append(float(controls["FAF"].get()))
        input_data.append(float(controls["TUE"].get()))

        # Convert to numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Predict using the loaded model
        prediction = lgbm_model.predict(input_array)

        # Display the prediction
        messagebox.showinfo("Prediction", f"Predicted obesity risk level: {prediction[0]}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Predict button
predict_button = ttk.Button(scrollable_frame, text="Predict", command=predict)
predict_button.pack(pady=20)

# Start the main event loop
root.mainloop()
